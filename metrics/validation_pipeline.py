"""
Validation pipeline for diffusion models.

This module implements a validation pipeline that handles:
1. Sampling from diffusion models
2. Computing metrics between generated samples and ground truth
3. Logging metrics and visualizations
4. Tracking model performance over time
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Callable, Optional, Any, Tuple, Union
import time
from dataclasses import dataclass

from metrics.image_metrics import compute_all_metrics


@dataclass
class ValidationConfig:
    """Configuration for validation pipeline."""
    num_validation_samples: int = 16
    diffusion_steps: int = 100  # Number of steps for sampling
    metrics_batch_size: int = 4  # Batch size for computing metrics
    log_every_n_steps: int = 1000  # Log metrics every N training steps
    save_best_model: bool = True  # Whether to save model with best metrics
    best_metric_key: str = "final_score"  # Metric to use for determining best model
    higher_is_better: bool = True  # Whether higher values are better for best_metric_key


class ValidationPipeline:
    """
    Validation pipeline for diffusion models.
    
    This class handles all validation-related operations including:
    - Generating samples
    - Computing metrics
    - Logging results
    - Tracking best models
    """
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize validation pipeline.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.best_score = -float('inf') if config.higher_is_better else float('inf')
        self.metrics_history = []
    
    def evaluate_samples(self, generated_samples: jnp.ndarray, ground_truth: jnp.ndarray) -> Dict[str, float]:
        """
        Compute metrics between generated samples and ground truth.
        
        Args:
            generated_samples: Generated samples (B, H, W, C)
            ground_truth: Ground truth samples (B, H, W, C)
            
        Returns:
            Dictionary of metrics
        """
        # Ensure values are in the expected range for metrics (usually 0-255)
        # We assume generated_samples are in [-1, 1] range from the diffusion model
        generated_samples_255 = ((generated_samples + 1) * 127.5).clip(0, 255)
        
        # If ground truth is already normalized from the dataset, convert to 0-255 range
        if ground_truth.max() <= 1.0:
            ground_truth_255 = (ground_truth * 255).clip(0, 255)
        else:
            ground_truth_255 = ground_truth.clip(0, 255)
            
        # Compute all metrics
        return compute_all_metrics(generated_samples_255, ground_truth_255)
    
    def is_better_score(self, current_score: float, best_score: float) -> bool:
        """Check if the current score is better than the best score."""
        if self.config.higher_is_better:
            return current_score > best_score
        else:
            return current_score < best_score
    
    def log_metrics(self, metrics: Dict[str, float], step: int, wandb=None) -> None:
        """
        Log metrics to console and wandb if available.
        
        Args:
            metrics: Dictionary of metrics
            step: Current training step
            wandb: Weights & Biases logger (optional)
        """
        # Print metrics to console
        print(f"\n--- Validation Metrics at Step {step} ---")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        # Log to wandb if available
        if wandb is not None:
            # Prefix metrics with 'val/' to distinguish from training metrics
            wandb_metrics = {f"val/{k}": v for k, v in metrics.items()}
            wandb.log(wandb_metrics, step=step)
        
        # Check if this is the best model so far
        current_score = metrics[self.config.best_metric_key]
        if self.is_better_score(current_score, self.best_score):
            self.best_score = current_score
            print(f"New best model! {self.config.best_metric_key}: {current_score:.4f}")
            
            # Log best score to wandb
            if wandb is not None:
                wandb.log({f"val/best_{self.config.best_metric_key}": current_score}, step=step)
                
            return True  # Signal that this is the best model
        
        return False  # Not the best model


def validate_diffusion_model(
    generate_samples_fn: Callable,
    validation_batch: Any,
    val_state: Any,
    pipeline: ValidationPipeline,
    current_step: int,
    wandb = None,
    diffusion_steps: int = None,
    save_model_fn: Callable = None
) -> Dict[str, float]:
    """
    Validate a diffusion model and compute metrics.
    
    Args:
        generate_samples_fn: Function to generate samples from the model
        validation_batch: Batch of validation data
        val_state: Model state for validation
        pipeline: ValidationPipeline instance
        current_step: Current training step
        wandb: Weights & Biases logger (optional)
        diffusion_steps: Number of diffusion steps (overrides config if provided)
        save_model_fn: Function to save the best model (optional)
        
    Returns:
        Dictionary of computed metrics
    """
    # Use provided diffusion steps or default from config
    steps = diffusion_steps if diffusion_steps is not None else pipeline.config.diffusion_steps
    
    # Get ground truth from validation batch (assuming standard data key)
    sample_data_key = "image" if "image" in validation_batch else list(validation_batch.keys())[0]
    ground_truth = validation_batch[sample_data_key]
    
    # Normalize ground truth to [-1, 1] if it's in [0, 255]
    if ground_truth.max() > 1.0:
        ground_truth = (ground_truth / 127.5) - 1.0
        
    # Generate samples
    start_time = time.time()
    generated_samples = generate_samples_fn(
        val_state=val_state,
        batch=validation_batch,
        sampler=val_state.sampler if hasattr(val_state, "sampler") else None,
        diffusion_steps=steps
    )
    generation_time = time.time() - start_time
    
    # Compute metrics
    metrics = pipeline.evaluate_samples(generated_samples, ground_truth)
    
    # Add generation time to metrics
    metrics['generation_time_sec'] = generation_time
    metrics['generation_time_per_sample'] = generation_time / len(generated_samples)
    
    # Log metrics
    is_best = pipeline.log_metrics(metrics, current_step, wandb)
    
    # Save best model if requested
    if is_best and pipeline.config.save_best_model and save_model_fn is not None:
        print(f"Saving best model with {pipeline.config.best_metric_key} = {metrics[pipeline.config.best_metric_key]:.4f}")
        save_model_fn(best=True)
    
    return metrics