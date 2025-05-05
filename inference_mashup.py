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

# Global job store
job_store = {}

# Default model name
DEFAULT_MODEL_NAME = 'diffusion-oxford_flowers102-res128-sweep-d4es07fm'

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

            samples = pipeline.generate_samples(
                num_samples=req.num_samples or len(req.prompts),
                resolution=req.resolution,
                diffusion_steps=req.diffusion_steps,
                guidance_scale=req.guidance_scale,
                start_step=0,
                conditioning_data=req.prompts,
            )
            images_b64 = []
            for img in samples:
                buf = io.BytesIO()
                img_np = np.array(img)
                img_uint8 = ((img_np + 1.0) * 127.5).astype("uint8")
                Image.fromarray(img_uint8).save(buf, format="PNG")
                images_b64.append(base64.b64encode(buf.getvalue()).decode())

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
