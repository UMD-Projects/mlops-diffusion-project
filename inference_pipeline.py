# %%
# Set JAX_PLATFORMS=''
import os
# os.environ["JAX_PLATFORMS"] = "cpu"
from flaxdiff.inference.pipeline import DiffusionInferencePipeline

# %%
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from wandb import Image as wandbImage
import tqdm
import grain.python as pygrain
import torch
from flaxdiff.samplers.euler import EulerAncestralSampler
import numpy as np

if __name__ == "__main__":
    # Parse the 'model registry' and 'version argument from the command line
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a model from the registry.')
    parser.add_argument('--model_registry', type=str, help='Model registry name', default="diffusion-laiona_coco-res256")
    parser.add_argument('--version', type=str, help='Model version', default='best')
    args = parser.parse_args()
    
    pipeline = DiffusionInferencePipeline.from_wandb_registry(
        modelname=args.model_registry,
        project='mlops-msml605-project',
        entity='umd-projects',
        version=args.version,
    )
    image_size = 256
    diffusion_steps = 200
    prompts = [
        'water tulip',
        'a water lily',
        'a water lily',
        'a photo of a rose',
        'a photo of a rose',
        'a water lily',
        'a water lily',
        'a photo of a marigold',
    ]
    generated = pipeline.generate_samples(
                num_samples=len(prompts),
                resolution=image_size,
                diffusion_steps=diffusion_steps,
                guidance_scale=3.0,
                start_step=1000,
                conditioning_data=prompts,
        )