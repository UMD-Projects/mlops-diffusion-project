"""
Image quality assessment metrics for diffusion model evaluation.

This module implements various image quality metrics including:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- FID (Fréchet Inception Distance)

These metrics help evaluate the quality of generated images compared to ground truth.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Union, List, Optional, Any
import functools

# Import metrics from established libraries
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
import torch
import lpips
import torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance

# Constants
MAX_PIXEL_VALUE = 255.0

# Initialize models (will be lazily loaded when needed)
_lpips_model = None
_fid_model = None

def get_lpips_model():
    """Get or initialize LPIPS model."""
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='alex')
    return _lpips_model

def get_fid_model(device='cpu'):
    """Get or initialize FID model."""
    global _fid_model
    if _fid_model is None:
        # Initialize with 2048 feature dimension and default InceptionV3
        _fid_model = FrechetInceptionDistance(feature=2048, normalize=True)
        _fid_model.to(device)
    return _fid_model

def psnr(img1: Union[jnp.ndarray, np.ndarray], 
         img2: Union[jnp.ndarray, np.ndarray], 
         max_val: float = MAX_PIXEL_VALUE) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images using scikit-image.
    
    PSNR is proportional to the logarithm of the reciprocal of MSE.
    Higher values indicate better quality.
    
    Args:
        img1: First image (typically generated) - shape (H, W, C)
        img2: Second image (typically ground truth) - shape (H, W, C)
        max_val: Maximum possible pixel value (255 for 8-bit images)
        
    Returns:
        PSNR value (higher is better)
    """
    # Convert to numpy if needed
    if isinstance(img1, jnp.ndarray):
        img1 = np.array(img1)
    if isinstance(img2, jnp.ndarray):
        img2 = np.array(img2)
        
    return sk_psnr(img1, img2, data_range=max_val)


def batch_psnr(batch_img1: Union[jnp.ndarray, np.ndarray], 
               batch_img2: Union[jnp.ndarray, np.ndarray], 
               max_val: float = MAX_PIXEL_VALUE) -> np.ndarray:
    """
    Calculate PSNR for a batch of images.
    
    Args:
        batch_img1: Batch of first images - shape (B, H, W, C)
        batch_img2: Batch of second images - shape (B, H, W, C)
        max_val: Maximum possible pixel value
        
    Returns:
        Array of PSNR values for each image pair
    """
    if isinstance(batch_img1, jnp.ndarray):
        batch_img1 = np.array(batch_img1)
    if isinstance(batch_img2, jnp.ndarray):
        batch_img2 = np.array(batch_img2)
        
    batch_size = batch_img1.shape[0]
    results = []
    
    for i in range(batch_size):
        results.append(psnr(batch_img1[i], batch_img2[i], max_val))
    
    return np.array(results)


def ssim(img1: Union[jnp.ndarray, np.ndarray], 
         img2: Union[jnp.ndarray, np.ndarray], 
         max_val: float = MAX_PIXEL_VALUE,
         multichannel: bool = True) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images using scikit-image.
    
    SSIM measures the similarity between two images based on luminance, contrast, and structure.
    Values range from -1 to 1, where 1 indicates perfect similarity.
    
    Args:
        img1: First image - shape (H, W, C)
        img2: Second image - shape (H, W, C)
        max_val: Maximum possible pixel value
        multichannel: Whether the images are multichannel (RGB, RGBA, etc.)
        
    Returns:
        SSIM value (higher is better)
    """
    # Convert to numpy if needed
    if isinstance(img1, jnp.ndarray):
        img1 = np.array(img1)
    if isinstance(img2, jnp.ndarray):
        img2 = np.array(img2)
    
    # scikit-image expects uint8 for multichannel images
    if multichannel:
        channel_axis = -1
    else:
        channel_axis = None
        
    return sk_ssim(img1, img2, 
                   data_range=max_val,
                   channel_axis=channel_axis)


def batch_ssim(batch_img1: Union[jnp.ndarray, np.ndarray], 
               batch_img2: Union[jnp.ndarray, np.ndarray], 
               max_val: float = MAX_PIXEL_VALUE) -> np.ndarray:
    """
    Calculate SSIM for a batch of images.
    
    Args:
        batch_img1: Batch of first images - shape (B, H, W, C)
        batch_img2: Batch of second images - shape (B, H, W, C)
        max_val: Maximum possible pixel value
        
    Returns:
        Array of SSIM values for each image pair
    """
    if isinstance(batch_img1, jnp.ndarray):
        batch_img1 = np.array(batch_img1)
    if isinstance(batch_img2, jnp.ndarray):
        batch_img2 = np.array(batch_img2)
        
    batch_size = batch_img1.shape[0]
    results = []
    
    for i in range(batch_size):
        results.append(ssim(batch_img1[i], batch_img2[i], max_val))
    
    return np.array(results)


def lpips_score(img1: Union[jnp.ndarray, np.ndarray], 
                img2: Union[jnp.ndarray, np.ndarray]) -> float:
    """
    Calculate Learned Perceptual Image Patch Similarity (LPIPS) using the official implementation.
    
    LPIPS uses deep features from pre-trained networks to measure perceptual similarity.
    Lower values indicate higher perceptual similarity.
    
    Args:
        img1: First image - shape (H, W, C)
        img2: Second image - shape (H, W, C)
        
    Returns:
        LPIPS score (lower is better)
    """
    # Convert to numpy if needed
    if isinstance(img1, jnp.ndarray):
        img1 = np.array(img1)
    if isinstance(img2, jnp.ndarray):
        img2 = np.array(img2)
    
    # Get the LPIPS model
    model = get_lpips_model()
    
    # LPIPS requires torch tensors in range [-1, 1]
    # Convert from [0, 255] to [-1, 1]
    img1_tensor = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    img2_tensor = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    
    with torch.no_grad():
        lpips_value = model(img1_tensor, img2_tensor).item()
    
    return lpips_value


def batch_lpips(batch_img1: Union[jnp.ndarray, np.ndarray], 
                batch_img2: Union[jnp.ndarray, np.ndarray]) -> np.ndarray:
    """
    Calculate LPIPS for a batch of images.
    
    Args:
        batch_img1: Batch of first images - shape (B, H, W, C)
        batch_img2: Batch of second images - shape (B, H, W, C)
        
    Returns:
        Array of LPIPS values for each image pair
    """
    if isinstance(batch_img1, jnp.ndarray):
        batch_img1 = np.array(batch_img1)
    if isinstance(batch_img2, jnp.ndarray):
        batch_img2 = np.array(batch_img2)
        
    batch_size = batch_img1.shape[0]
    results = []
    
    # Note: A more optimized approach would batch process all images
    # but this is clearer for illustration
    for i in range(batch_size):
        results.append(lpips_score(batch_img1[i], batch_img2[i]))
    
    return np.array(results)


def compute_fid(real_images: Union[jnp.ndarray, np.ndarray], 
               generated_images: Union[jnp.ndarray, np.ndarray],
               device='cpu') -> float:
    """
    Calculate Fréchet Inception Distance (FID) between real and generated images.
    
    Note: FID is most accurate with large batches (100+ images).
    This function will compute FID with whatever is provided, but small batches
    may have less statistical significance.
    
    Args:
        real_images: Real reference images - shape (B, H, W, C) with values in [0, 255]
        generated_images: Generated images - shape (B, H, W, C) with values in [0, 255]
        device: Computation device ('cpu' or 'cuda')
        
    Returns:
        FID score (lower is better)
    """
    # Convert to numpy if needed
    if isinstance(real_images, jnp.ndarray):
        real_images = np.array(real_images)
    if isinstance(generated_images, jnp.ndarray):
        generated_images = np.array(generated_images)
    
    # FID expects uint8 images
    real_images = real_images.astype(np.uint8)
    generated_images = generated_images.astype(np.uint8)
    
    # Get the FID model
    fid_model = get_fid_model(device)
    
    # Convert to PyTorch tensors - FID expects NCHW format and uint8
    # Input is NHWC, so we need to transpose
    real_tensor = torch.from_numpy(real_images).permute(0, 3, 1, 2)
    gen_tensor = torch.from_numpy(generated_images).permute(0, 3, 1, 2)
    
    # Reset FID model internal state
    fid_model.reset()
    
    # Update with real and generated images
    fid_model.update(real_tensor, real=True)
    fid_model.update(gen_tensor, real=False)
    
    # Compute FID
    try:
        fid_value = fid_model.compute().item()
        return fid_value
    except Exception as e:
        print(f"Warning: FID computation failed - {str(e)}")
        print("This is often due to insufficient batch size. FID is most meaningful with 100+ images.")
        return float('nan')


def compute_all_metrics(generated_images: Union[jnp.ndarray, np.ndarray], 
                       real_images: Union[jnp.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Compute all available metrics between generated and real images.
    
    Args:
        generated_images: Generated images - shape (B, H, W, C)
        real_images: Real/reference images - shape (B, H, W, C)
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Convert to numpy if needed
    if isinstance(generated_images, jnp.ndarray):
        generated_images = np.array(generated_images)
    if isinstance(real_images, jnp.ndarray):
        real_images = np.array(real_images)
    
    # Compute PSNR
    psnr_values = batch_psnr(generated_images, real_images)
    metrics['psnr'] = float(np.mean(psnr_values))
    
    # Compute SSIM
    ssim_values = batch_ssim(generated_images, real_images)
    metrics['ssim'] = float(np.mean(ssim_values))
    
    # Compute LPIPS (perceptual similarity) - can be computationally expensive
    try:
        lpips_values = batch_lpips(generated_images, real_images)
        metrics['lpips'] = float(np.mean(lpips_values))
    except Exception as e:
        print(f"Warning: LPIPS computation failed - {str(e)}")
        metrics['lpips'] = 1.0  # Default to worst value on failure
    
    # Compute FID if the batch is large enough (at least 2 images)
    # For very small batches, FID might not be meaningful but we'll compute it anyway
    if generated_images.shape[0] >= 2:
        try:
            metrics['fid'] = compute_fid(real_images, generated_images)
        except Exception as e:
            print(f"Warning: FID computation failed - {str(e)}")
            metrics['fid'] = 100.0  # Default to a poor (but not NaN) value on failure
    else:
        print("Warning: FID requires at least 2 images. Using default value.")
        metrics['fid'] = 100.0  # Default to a poor (but not NaN) value
    
    # Compute final score (weighted combination)
    # Higher is better, scale is 0-10
    metrics['final_score'] = compute_final_score(metrics)
    
    return metrics


def compute_final_score(metrics: Dict[str, float]) -> float:
    """
    Compute a final quality score from individual metrics.
    
    The final score is a weighted combination of metrics, normalized
    to a 0-10 scale where higher values indicate better quality.
    
    Args:
        metrics: Dictionary with individual metrics
        
    Returns:
        Final quality score (0-10 scale, higher is better)
    """
    # Define weights for different metrics
    # These weights should be tuned based on domain knowledge
    weights = {
        'psnr': 0.3,    # PSNR (higher is better)
        'ssim': 0.3,    # SSIM (higher is better)
        'lpips': -0.2,  # LPIPS (lower is better, hence negative weight)
        'fid': -0.2     # FID (lower is better, hence negative weight)
    }
    
    # Normalize metrics to 0-1 scale based on typical ranges
    normalized = {}
    
    # PSNR typically ranges from ~10-40 for reasonable images
    normalized['psnr'] = min(max(metrics['psnr'] - 20, 0) / 20, 1)
    
    # SSIM is already in 0-1 range
    normalized['ssim'] = max(min(metrics['ssim'], 1), 0)
    
    # LPIPS is typically in 0-1 range (lower is better)
    if 'lpips' in metrics and not np.isnan(metrics['lpips']):
        normalized['lpips'] = max(1 - metrics['lpips'], 0)
    else:
        normalized['lpips'] = 0  # Worst case if LPIPS is NaN
    
    # FID normalization (typical range might be 0-200)
    if 'fid' in metrics and not np.isnan(metrics['fid']):
        # FID can range widely, but usually 0-200 is a reasonable range
        normalized['fid'] = max(1 - metrics['fid'] / 100, 0)
    else:
        normalized['fid'] = 0  # Worst case if FID is NaN
    
    # Compute weighted sum of available metrics
    score = 0
    total_weight = 0
    
    for metric, weight in weights.items():
        if metric in normalized:
            score += normalized[metric] * weight
            total_weight += abs(weight)
    
    # Scale to 0-10
    if total_weight > 0:
        final_score = 10 * max(0, min(score / total_weight, 1))
    else:
        # Fallback if no metrics were available
        final_score = 0
    
    return final_score


class FIDAccumulator:
    """
    FID Accumulator for tracking FID over multiple batches.
    
    This allows computing FID over many small batches by accumulating images,
    which is useful during training when you can't easily get hundreds of
    images in a single batch.
    """
    def __init__(self, device='cpu'):
        """Initialize the FID accumulator."""
        self.fid_model = FrechetInceptionDistance(feature=2048, normalize=True)
        self.fid_model.to(device)
        self.device = device
        self.real_count = 0
        self.gen_count = 0
        self.reset()
    
    def reset(self):
        """Reset the accumulated statistics."""
        self.fid_model.reset()
        self.real_count = 0
        self.gen_count = 0
    
    def update(self, images, real=True):
        """
        Add a batch of images to the accumulator.
        
        Args:
            images: Image batch of shape [B,H,W,C] with values in [0,255]
            real: Whether these are real images (True) or generated (False)
        """
        # Convert to numpy if needed
        if isinstance(images, jnp.ndarray):
            images = np.array(images)
        
        # Ensure uint8 type
        images = images.astype(np.uint8)
        
        # Convert to PyTorch tensor and transpose to NCHW format
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
        
        # Update the model
        self.fid_model.update(tensor, real=real)
        
        # Update counters
        if real:
            self.real_count += images.shape[0]
        else:
            self.gen_count += images.shape[0]
    
    def compute(self):
        """
        Compute the FID score using all accumulated images.
        
        Returns:
            FID score (lower is better) or NaN if not enough images
        """
        try:
            # We need at least a few images on both sides
            if self.real_count < 2 or self.gen_count < 2:
                print(f"Warning: Not enough images for FID. Real: {self.real_count}, Gen: {self.gen_count}")
                return float('nan')
            
            # Compute FID
            return self.fid_model.compute().item()
        except Exception as e:
            print(f"FID computation failed: {str(e)}")
            return float('nan')