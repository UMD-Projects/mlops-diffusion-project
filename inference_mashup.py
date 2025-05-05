import os
import uuid
import base64
import io
import threading
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from flaxdiff.inference.pipeline import DiffusionInferencePipeline
from PIL import Image
import numpy as np

# Force JAX to CPU
os.environ["JAX_PLATFORMS"] = os.getenv("JAX_PLATFORMS", "cpu")

app = FastAPI()
job_store = {}

DEFAULT_MODEL_NAME = 'diffusion-laiona_coco-res256'

class GenerateRequest(BaseModel):
    prompts: List[str]
    model_name: Optional[str] = None
    num_samples: Optional[int] = 1
    resolution: Optional[int] = 64
    diffusion_steps: Optional[int] = 25
    guidance_scale: Optional[float] = 3.0

@app.post("/generate")
def generate(req: GenerateRequest):
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "running", "result": None}

    def run_generation():
        try:
            model_name = req.model_name or DEFAULT_MODEL_NAME
            print(f"[Job {job_id}] Loading model: {model_name}")
            pipeline = DiffusionInferencePipeline.from_wandb_registry(
                modelname=model_name,
                project='wandb-registry-model',
                entity='umd-projects'
            )
            print(f"[Job {job_id}] Model loaded")
            print(f"[Job {job_id}] Input config: {pipeline.config.get('input_config', {})}")

            # Fallback if autoencoder is missing
            if not hasattr(pipeline.autoencoder, 'decode'):
                print(f"[Job {job_id}] Falling back to hardcoded Hugging Face VAE: pcuenq/sd-vae-ft-mse-flax")
                from flaxdiff.models.autoencoder.diffusers import StableDiffusionVAE
                try:
                    pipeline.autoencoder = StableDiffusionVAE(modelname="pcuenq/sd-vae-ft-mse-flax")
                    print(f"[Job {job_id}] Hugging Face VAE loaded successfully.")
                except Exception as e:
                    raise RuntimeError(f"Failed to load hardcoded VAE: {e}")

            # Run sampling
            samples = pipeline.generate_samples(
                num_samples=req.num_samples or len(req.prompts),
                resolution=req.resolution,
                diffusion_steps=req.diffusion_steps,
                guidance_scale=req.guidance_scale,
                start_step=1000,
                conditioning_data=req.prompts,
            )

            images_b64 = []
            for i, img in enumerate(samples):
                buf = io.BytesIO()
                try:
                    img_np = np.array(img) if not isinstance(img, np.ndarray) else img
                    if img_np.ndim == 2:
                        img_np = np.stack([img_np]*3, axis=-1)

                    img_np = np.clip(img_np, -1.0, 1.0)
                    img_uint8 = ((img_np + 1.0) * 127.5).astype("uint8")
                    img_uint8 = np.nan_to_num(img_uint8, nan=0).astype("uint8")

                    image = Image.fromarray(img_uint8)
                    image.save(buf, format="PNG")
                    images_b64.append(base64.b64encode(buf.getvalue()).decode())
                except Exception as e:
                    print(f"[Job {job_id}] Failed to process image {i}: {e}")

            if not images_b64:
                raise RuntimeError("All images failed to process.")

            job_store[job_id] = {"status": "completed", "result": images_b64}
            print(f"[Job {job_id}] Generation complete")

        except Exception as e:
            job_store[job_id] = {"status": "failed", "error": str(e)}
            print(f"[Job {job_id}] Generation failed: {e}")

    threading.Thread(target=run_generation).start()
    return {"job_id": job_id, "status": "running"}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **job}
