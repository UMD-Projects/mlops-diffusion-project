import os
import uuid
import base64
import io
import json
import threading
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np
import jax

from flaxdiff.utils import parse_config, RandomMarkovState
from flaxdiff.inference.pipeline import DiffusionInferencePipeline

# Force JAX to CPU
os.environ["JAX_PLATFORMS"] = os.getenv("JAX_PLATFORMS", "cpu")

app = FastAPI()
job_store = {}

CONFIG_PATH = "config.json"

class GenerateRequest(BaseModel):
    prompts: List[str]
    num_samples: Optional[int] = 1
    resolution: Optional[int] = 256
    diffusion_steps: Optional[int] = 25
    guidance_scale: Optional[float] = 3.0

# Load model once at startup
print("[Startup] Loading config...")
with open(CONFIG_PATH, "r") as f:
    raw_config = json.load(f)
parsed_config = parse_config(raw_config)

# You need to plug in actual model weights here
# Replace with real loaded weights from WandB or other source
dummy_state = {"params": {}, "ema_params": {}}

pipeline = DiffusionInferencePipeline.create(
    config=parsed_config,
    state=dummy_state,
    best_state=dummy_state,
    rngstate=RandomMarkovState(jax.random.PRNGKey(42))
)
print("[Startup] Model initialized successfully")

@app.post("/generate")
def generate(req: GenerateRequest):
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "running", "result": None}

    def run_generation():
        try:
            print(f"[Job {job_id}] Generating with prompts: {req.prompts}")

            samples = pipeline.generate_samples(
                num_samples=req.num_samples or len(req.prompts),
                resolution=req.resolution,
                diffusion_steps=req.diffusion_steps,
                guidance_scale=req.guidance_scale,
                conditioning_data=req.prompts
            )

            images_b64 = []
            for i, img in enumerate(samples):
                buf = io.BytesIO()
                try:
                    img_np = np.array(img) if not isinstance(img, np.ndarray) else img
                    if img_np.ndim == 2:
                        img_np = np.stack([img_np] * 3, axis=-1)

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
