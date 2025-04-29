"""
Metrics package for evaluating diffusion models.

This package provides tools for evaluating the quality of generated images and videos
from diffusion models, including standard metrics like PSNR, SSIM, and a validation pipeline.
"""

from metrics.image_metrics import (
    psnr, batch_psnr, 
    ssim, batch_ssim, 
    lpips_score, batch_lpips,
    compute_all_metrics,
    compute_final_score
)

from metrics.validation_pipeline import (
    ValidationConfig,
    ValidationPipeline,
    validate_diffusion_model
)

__all__ = [
    # Image metrics
    'psnr', 'batch_psnr',
    'ssim', 'batch_ssim',
    'lpips_score', 'batch_lpips',
    'compute_all_metrics', 'compute_final_score',
    
    # Validation pipeline
    'ValidationConfig', 'ValidationPipeline', 'validate_diffusion_model'
]