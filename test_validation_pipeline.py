#!/usr/bin/env python3
"""
Test script for validating the metrics and validation pipeline implementation.

This script:
1. Creates dummy generated and ground truth images
2. Tests each individual metric function
3. Tests the combined metrics computation
4. Tests the validation pipeline with simple data
"""

# Set JAX to use CPU before importing JAX
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import time
from typing import Dict
import sys

# Add the project directory to the path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our metrics module
from metrics import (
    psnr, ssim, lpips_simple, fid_simple, 
    compute_all_metrics, compute_final_score,
    ValidationConfig, ValidationPipeline
)

# Print JAX config for debugging
print(f"JAX is using: {jax.devices()}")

def create_test_images(batch_size=4, height=64, width=64, channels=3):
    """Create synthetic test images for metrics evaluation."""
    
    # Create a key for random number generation
    key = jax.random.PRNGKey(42)
    
    # Generate "ground truth" images (simulating real images)
    key, subkey = jax.random.split(key)
    real_images = jax.random.uniform(
        subkey, shape=(batch_size, height, width, channels), 
        minval=0, maxval=255
    )
    
    # Generate "predicted" images with controlled quality
    # We'll add noise to simulate different levels of quality
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, shape=real_images.shape) * 30.0
    generated_images = jnp.clip(real_images + noise, 0, 255)
    
    # Create a second set with more noise (worse quality)
    key, subkey = jax.random.split(key)
    more_noise = jax.random.normal(subkey, shape=real_images.shape) * 70.0
    worse_images = jnp.clip(real_images + more_noise, 0, 255)
    
    return real_images, generated_images, worse_images


def test_individual_metrics():
    """Test each individual metric function."""
    print("\n=== Testing Individual Metrics ===")
    
    # Create test images
    real_images, generated_images, worse_images = create_test_images()
    
    # Test PSNR
    psnr_value = psnr(generated_images[0], real_images[0])
    worse_psnr = psnr(worse_images[0], real_images[0])
    print(f"PSNR (higher is better): {psnr_value:.4f} vs {worse_psnr:.4f} (worse quality)")
    assert psnr_value > worse_psnr, "PSNR should be higher for better quality images"
    
    # Test SSIM
    ssim_value = ssim(generated_images[0], real_images[0])
    worse_ssim = ssim(worse_images[0], real_images[0])
    print(f"SSIM (higher is better): {ssim_value:.4f} vs {worse_ssim:.4f} (worse quality)")
    assert ssim_value > worse_ssim, "SSIM should be higher for better quality images"
    
    # Test LPIPS
    lpips_value = lpips_simple(generated_images[0], real_images[0])
    worse_lpips = lpips_simple(worse_images[0], real_images[0])
    print(f"LPIPS (lower is better): {lpips_value:.4f} vs {worse_lpips:.4f} (worse quality)")
    assert lpips_value < worse_lpips, "LPIPS should be lower for better quality images"
    
    # Test FID
    fid_value = fid_simple(real_images, generated_images)
    worse_fid = fid_simple(real_images, worse_images)
    print(f"FID (lower is better): {fid_value:.4f} vs {worse_fid:.4f} (worse quality)")
    assert fid_value < worse_fid, "FID should be lower for better quality images"
    
    print("All individual metrics tests passed!")


def test_combined_metrics():
    """Test the combined metrics computation."""
    print("\n=== Testing Combined Metrics ===")
    
    # Create test images
    real_images, generated_images, worse_images = create_test_images()
    
    # Compute metrics for both sets
    metrics = compute_all_metrics(generated_images, real_images)
    worse_metrics = compute_all_metrics(worse_images, real_images)
    
    print("\nBetter quality metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
        
    print("\nWorse quality metrics:")
    for k, v in worse_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Final score should be higher for better quality
    assert metrics['final_score'] > worse_metrics['final_score'], \
        "Final score should be higher for better quality images"
    
    print("Combined metrics test passed!")


def test_validation_pipeline():
    """Test the validation pipeline."""
    print("\n=== Testing Validation Pipeline ===")
    
    # Create test images and a mock validation batch
    real_images, generated_images, _ = create_test_images()
    validation_batch = {"image": real_images}
    
    # Create a validation configuration
    val_config = ValidationConfig(
        num_validation_samples=4,
        diffusion_steps=100,
        save_best_model=True
    )
    
    # Initialize the validation pipeline
    pipeline = ValidationPipeline(val_config)
    
    # Track the best score before evaluation
    initial_best_score = pipeline.best_score
    
    # Evaluate the samples
    metrics = pipeline.evaluate_samples(generated_images, real_images)
    
    # Log the metrics
    is_best = pipeline.log_metrics(metrics, step=1000)
    
    # Should be the best since it's the first evaluation
    assert is_best, "First evaluation should be the best"
    assert pipeline.best_score > initial_best_score, "Best score should be updated"
    
    # Test with another set of images (worse quality)
    _, _, worse_images = create_test_images(batch_size=4)
    worse_metrics = pipeline.evaluate_samples(worse_images, real_images)
    
    # Log the metrics again
    is_best = pipeline.log_metrics(worse_metrics, step=2000)
    
    # Should not be the best since quality is worse
    assert not is_best, "Worse quality should not update best score"
    
    print("Validation pipeline test passed!")


def test_mock_diffusion_validation():
    """Test the validation pipeline with a mock diffusion model."""
    print("\n=== Testing Mock Diffusion Validation ===")
    
    # Create a validation configuration
    val_config = ValidationConfig(
        num_validation_samples=4,
        diffusion_steps=100,
        save_best_model=True
    )
    
    # Initialize the validation pipeline
    pipeline = ValidationPipeline(val_config)
    
    # Create mock real images
    batch_size = 4
    height, width, channels = 64, 64, 3
    real_images = jnp.ones((batch_size, height, width, channels)) * 127.5
    validation_batch = {"image": real_images}
    
    # Mock state and generate_samples_fn for the validation function
    class MockState:
        def __init__(self):
            self.sampler = None

    # Create a mock function that simulates generating samples
    def mock_generate_samples(val_state, batch, sampler=None, diffusion_steps=None):
        # Just add some noise to the real images as a simulation
        key = jax.random.PRNGKey(int(time.time()))
        noise = jax.random.normal(key, shape=batch["image"].shape) * 20.0
        return jnp.clip(batch["image"] + noise, 0, 255) 

    # Create a mock save function
    def mock_save_fn(best=False):
        print(f"Model {'would be' if best else 'would not be'} saved as best!")
    
    # Call the validate_diffusion_model function (imported from our validation_pipeline module)
    from metrics.validation_pipeline import validate_diffusion_model
    
    mock_state = MockState()
    metrics = validate_diffusion_model(
        generate_samples_fn=mock_generate_samples,
        validation_batch=validation_batch,
        val_state=mock_state,
        pipeline=pipeline,
        current_step=1000,
        save_model_fn=mock_save_fn
    )
    
    # Verify we got metrics back
    assert 'psnr' in metrics, "PSNR should be in returned metrics"
    assert 'ssim' in metrics, "SSIM should be in returned metrics"
    assert 'final_score' in metrics, "Final score should be in returned metrics"
    
    print("Mock diffusion validation test passed!")


if __name__ == "__main__":
    print("Starting validation pipeline tests...")
    
    # Run all tests
    test_individual_metrics()
    test_combined_metrics()
    test_validation_pipeline()
    test_mock_diffusion_validation()
    
    print("\n✅ All tests passed! The validation pipeline is working as expected.")