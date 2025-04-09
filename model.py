import base64
import io
import numpy as np
from datetime import datetime
import jax
import jax.numpy as jnp
import optax
from PIL import Image

from flaxdiff.schedulers import EDMNoiseScheduler, KarrasVENoiseScheduler
from flaxdiff.predictors import KarrasPredictionTransform
from flaxdiff.models.simple_unet import Unet
from flaxdiff.trainer import DiffusionTrainer
from flaxdiff.utils import defaultTextEncodeModel
from flaxdiff.samplers.euler import EulerAncestralSampler

# --- Global Configuration ---
IMAGE_SIZE = 128

# --- Utility Function ---
def jax_array_to_base64(sample: jnp.ndarray) -> str:
    np_img = np.array(sample)
    np_img = np.clip((np_img + 1.0) * 127.5, 0, 255).astype(np.uint8)
    im = Image.fromarray(np_img)
    buffered = io.BytesIO()
    im.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- Load Model Once on Startup ---
def load_model():
    text_encoder = defaultTextEncodeModel()

    edm_schedule = EDMNoiseScheduler(1, sigma_max=80, rho=7, sigma_data=0.5)
    karas_ve_schedule = KarrasVENoiseScheduler(1, sigma_max=80, rho=7, sigma_data=0.5)

    input_shapes = {
        "x": (IMAGE_SIZE, IMAGE_SIZE, 3),
        "temb": (),
        "textcontext": (77, 768)
    }

    unet = Unet(
        emb_features=256,
        feature_depths=[64, 64, 128, 256, 512],
        attention_configs=[
            None,
            {"heads": 8, "dtype": jnp.float32, "flash_attention": False, "use_projection": False, "use_self_and_cross": True},
            {"heads": 8, "dtype": jnp.float32, "flash_attention": False, "use_projection": False, "use_self_and_cross": True},
            {"heads": 8, "dtype": jnp.float32, "flash_attention": False, "use_projection": False, "use_self_and_cross": True},
            {"heads": 8, "dtype": jnp.float32, "flash_attention": False, "use_projection": False, "use_self_and_cross": False}
        ],
        num_res_blocks=2,
        num_middle_res_blocks=1
    )

    solver = optax.adam(2e-4)

    checkpoint_path = "/app/checkpoints/diffusion_sde_ve_2025-04-08_16:00:25"

    trainer = DiffusionTrainer(
        unet,
        optimizer=solver,
        input_shapes=input_shapes,
        noise_schedule=edm_schedule,
        rngs=jax.random.PRNGKey(4),
        name="inference",
        model_output_transform=KarrasPredictionTransform(sigma_data=edm_schedule.sigma_data),
        encoder=text_encoder,
        distributed_training=True,
        wandb_config=None,
        load_from_checkpoint=checkpoint_path
    )

    null_labels_full = text_encoder([""])

    sampler = EulerAncestralSampler(
        trainer.model,
        None,
        noise_schedule=karas_ve_schedule,
        image_size=IMAGE_SIZE,
        autoencoder=trainer.autoencoder,
        model_output_transform=trainer.model_output_transform,
        guidance_scale=4,
        null_labels_seq=null_labels_full
    )

    return {
        "trainer": trainer,
        "text_encoder": text_encoder,
        "sampler": sampler
    }