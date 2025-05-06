from flaxdiff.metrics.common import EvaluationMetric
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Callable, Dict, Any, Tuple, Optional
import os
from flax.core import FrozenDict
    
def get_clip_metric(
    modelname: str = "openai/clip-vit-large-patch14",
    clipmodel = None,
    processor = None,
):
    from transformers import AutoProcessor, FlaxCLIPModel
    model = FlaxCLIPModel.from_pretrained(modelname, dtype=jnp.float16) if clipmodel is None else clipmodel
    processor = AutoProcessor.from_pretrained(modelname, use_fast=False, dtype=jnp.float16) if processor is None else processor
    
    @jax.jit
    def calc(pixel_values, input_ids, attention_mask):
        # Get the logits
        generated_out = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
            
        gen_img_emb = generated_out.image_embeds
        txt_emb = generated_out.text_embeds

        # 1. Normalize embeddings (essential for cosine similarity/distance)
        gen_img_emb = gen_img_emb / (jnp.linalg.norm(gen_img_emb, axis=-1, keepdims=True) + 1e-6)
        txt_emb = txt_emb / (jnp.linalg.norm(txt_emb, axis=-1, keepdims=True) + 1e-6)

        # 2. Calculate cosine similarity
        # Using einsum for batch dot product: batch (b), embedding_dim (d) -> bd,bd->b
        # Calculate cosine similarity
        similarity = jnp.einsum('bd,bd->b', gen_img_emb, txt_emb)

        scaled_distance = (1.0 - similarity)
        # 4. Average over the batch
        mean_scaled_distance = jnp.mean(scaled_distance)

        return mean_scaled_distance
        
    def clip_metric(
        generated: jnp.ndarray,
        batch
    ):
        original_conditions = batch['text']
        
        # Convert samples from [-1, 1] to [0, 255] and uint8
        generated = (((generated + 1.0) / 2.0) * 255).astype(jnp.uint8)
        
        generated_inputs = processor(images=generated, return_tensors="jax", padding=True,)
        
        pixel_values = generated_inputs['pixel_values']
        input_ids = original_conditions['input_ids']
        attention_mask = original_conditions['attention_mask']
        
        return calc(pixel_values, input_ids, attention_mask)
    
    return EvaluationMetric(
        function=clip_metric,
        name='clip_similarity'
    )
    


def get_kcfd_metric(
    real_features_path: str, # Path to precomputed CLIP features of real images
    clip_modelname: str = "openai/clip-vit-large-patch14",
    degree: int = 3,
    gamma: Optional[float] = None, # Default: 1 / embedding_dim
    coef0: float = 1.0,
    clipmodel = None,
    processor = None,
):
    """
    Creates a batch-wise Kernel CLIP Feature Distance (KCFD) metric (MMD^2).

    NOTE: This is NOT standard KID. It uses CLIP image features instead of
    InceptionV3 features. It computes MMD^2 between the current batch
    of generated CLIP features and a random subset of pre-computed real
    CLIP features of the same size. The score will be noisy due to small
    batch size. Lower is generally better.

    Args:
        real_features_path: Path to pre-computed real CLIP image features
                            (.npy file, shape [N, embedding_dim]).
        clip_modelname: The Hugging Face identifier for the CLIP model.
        degree: Degree for the polynomial kernel.
        gamma: Gamma for the polynomial kernel. Defaults to 1.0 / embedding_dim.
        coef0: Coef0 for the polynomial kernel.

    Returns:
        An EvaluationMetric instance for the batch-wise KCFD score.

    Raises:
        FileNotFoundError: If real_features_path does not exist.
        ValueError: If real_features have incorrect shape or dimension mismatch.
        ImportError: If transformers/flax/jax/numpy are not installed.
    """
    # Imports within function scope for encapsulation
    from transformers import AutoProcessor, FlaxCLIPModel

    # --- Load Model and Real Features (once per metric creation) ---
    try:
        print(f"KCFD: Loading CLIP model ({clip_modelname})...")
        # Load CLIP model and processor - use float32 for stability in MMD
        model = FlaxCLIPModel.from_pretrained(clip_modelname, dtype=jnp.float32) if clipmodel is None else clipmodel
        processor = AutoProcessor.from_pretrained(clip_modelname) if processor is None else processor
        params = model.params
        embedding_dim = model.config.vision_config.hidden_size

        print(f"KCFD: Loading real CLIP features from {real_features_path}...")
        real_features = jnp.load(real_features_path).astype(jnp.float32)
        if real_features.ndim != 2 or real_features.shape[1] != embedding_dim:
            raise ValueError(
                f"Real CLIP features shape mismatch. Expected [N, {embedding_dim}], "
                f"got {real_features.shape}"
            )
        if real_features.shape[0] < 2:
            raise ValueError("Need at least 2 real feature samples for MMD calculation.")
        num_real_features = real_features.shape[0]
        print(f"KCFD: Loaded {num_real_features} real features with dim {embedding_dim}.")

    except Exception as e:
        print(f"KCFD: Failed to load model or features: {e}")
        raise

    # --- Internal Helper Functions (JITted) ---
    final_gamma = gamma if gamma is not None else (1.0 / embedding_dim)

    @partial(jax.jit) # Preprocessing can often be JITted
    def _preprocess_for_clip(images_neg1_1: jnp.ndarray) -> jnp.ndarray:
        """Preprocesses images from [-1, 1] range for CLIP's image encoder."""
        # Convert JAX array to NumPy for processor - Processor usually expects this
        # or PIL images. This step might break JIT if processor isn't JAX-native.
        # A pure-JAX preprocessing might be needed if JIT fails here.
        # For now, assume this preprocessing step happens outside main calc JIT.
        # Let's redefine preprocessing to be pure JAX based on typical CLIP needs.

        # 1. Map [-1, 1] to [0, 1]
        images_0_1 = (images_neg1_1.astype(jnp.float32) + 1.0) / 2.0

        # 2. Resize (Example: to 224x224 using bicubic) - get size from processor
        target_size = (
            processor.image_processor.crop_size['height'],
            processor.image_processor.crop_size['width']
        )
        resized = jax.image.resize(images_0_1,
                                   (images_neg1_1.shape[0], *target_size, images_neg1_1.shape[-1]),
                                   method='bicubic') # Or 'bilinear'

        # 3. Normalize using CLIP's mean and std
        mean = jnp.array(processor.image_processor.image_mean, dtype=jnp.float32).reshape(1, 1, 1, -1)
        std = jnp.array(processor.image_processor.image_std, dtype=jnp.float32).reshape(1, 1, 1, -1)
        normalized = (resized - mean) / (std + 1e-6)

        # 4. Transpose to B, C, H, W
        pixel_values = jnp.transpose(normalized, (0, 3, 1, 2))
        return pixel_values.astype(jnp.float32) # Ensure float32

    @partial(jax.jit, static_argnames=['apply_fn'])
    def _extract_clip_features(apply_fn: Callable, params: FrozenDict, pixel_values: jnp.ndarray) -> jnp.ndarray:
        """Extracts CLIP image features."""
        # Pass pixel_values positionally
        features = apply_fn(
            {'params': params},
            pixel_values,
            method=model.get_image_features # Use actual method reference
            )
        return features # Shape [B, embedding_dim]

    @partial(jax.jit, static_argnums=(2,)) # degree is static
    def _polynomial_kernel(X: jnp.ndarray, Y: jnp.ndarray, degree: int) -> jnp.ndarray:
        """Computes polynomial kernel using gamma/coef0 from outer scope."""
        # Use final_gamma, coef0 captured from the outer scope
        return jnp.power((final_gamma * (X @ Y.T)) + coef0, degree)

    @partial(jax.jit, static_argnums=(2,)) # degree is static
    def _compute_mmd2(X: jnp.ndarray, Y: jnp.ndarray, degree: int) -> jnp.ndarray:
        """Computes unbiased MMD^2 based on polynomial kernel."""
        n, m = X.shape[0], Y.shape[0]
        # MMD is undefined for fewer than 2 samples in either set
        if n < 2 or m < 2: return jnp.array(jnp.nan, dtype=jnp.float32)

        Kxx = _polynomial_kernel(X, X, degree)
        Kyy = _polynomial_kernel(Y, Y, degree)
        Kxy = _polynomial_kernel(X, Y, degree)

        # Ensure diagonal terms are handled correctly for n=1 or m=1 edge cases (though prevented above)
        term1 = (jnp.sum(Kxx) - jnp.trace(Kxx)) / (n * (n - 1))
        term2 = (jnp.sum(Kyy) - jnp.trace(Kyy)) / (m * (m - 1))
        term3 = (2 / (n * m)) * jnp.sum(Kxy)

        # Clip at 0 to prevent negative values due to numerical estimation variance
        return jnp.maximum(0.0, term1 + term2 - term3)

    # --- The Metric Function (returned by factory) ---
    def kcfd_metric(generated: jnp.ndarray, batch: Dict) -> jnp.ndarray:
        """Calculates batch-wise KCFD (MMD^2 on CLIP features). Lower is better."""
        # batch argument is unused for this metric, only needs generated samples
        batch_size = generated.shape[0]
        if batch_size < 2: return jnp.array(jnp.nan, dtype=jnp.float32) # MMD undefined

        # Preprocess and Extract features for the generated batch
        # Preprocessing is pure JAX, can be included in JIT or called before
        pixel_values = _preprocess_for_clip(generated)
        gen_features = _extract_clip_features(model.apply, params, pixel_values) # Pass model's apply method

        # Sample real features (using numpy for simpler index sampling outside JIT)
        effective_bs = min(batch_size, num_real_features)
        if effective_bs < 2: return jnp.array(jnp.nan, dtype=jnp.float32) # Need at least 2 samples

        # Trim gen_features if we have fewer real features than the batch size
        if effective_bs < batch_size:
            gen_features = gen_features[:effective_bs]

        # Sample indices using NumPy random - practical tradeoff for encapsulation
        indices = np.random.choice(num_real_features, effective_bs, replace=False)
        # Use JAX array for indexing into JAX real_features array
        sampled_real_features = real_features[jnp.array(indices)]

        # Calculate MMD^2 score using the JITted function
        mmd2_score = _compute_mmd2(gen_features, sampled_real_features, degree)

        return mmd2_score.astype(jnp.float32) # Ensure consistent output type

    # --- Return EvaluationMetric ---
    return EvaluationMetric(
        function=kcfd_metric,
        name='kcfd_batch_mmd2' # Kernel CLIP Feature Distance (batch MMD^2)
    )