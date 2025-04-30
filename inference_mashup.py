import os
os.environ["JAX_PLATFORMS"] = "cpu"

# inference.py


from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from flaxdiff.inference.pipeline import DiffusionInferencePipeline
import base64, io
from PIL import Image

# Force JAX to CPU (if you have GPU, adjust/remove)
os.environ["JAX_PLATFORMS"] = os.getenv("JAX_PLATFORMS", "cpu")

# # Initialize diffusion pipeline from W&B registry
# pipeline = DiffusionInferencePipeline.from_wandb_registry(
#     modelname='diffusion-oxford_flowers102-res256',
#     project='mlops-msml605-project',
#     entity='umd-projects',
# )

# # Request schema
# class GenerateRequest(BaseModel):
#     prompts: List[str]
#     num_samples: Optional[int] = None
#     resolution: int = 256
#     diffusion_steps: int = 100
#     guidance_scale: float = 3.0

# app = FastAPI()

# # Inference endpoint
# @app.post("/generate")
# def generate(req: GenerateRequest):
#     samples = pipeline.generate_samples(
#         num_samples=req.num_samples or len(req.prompts),
#         resolution=req.resolution,
#         diffusion_steps=req.diffusion_steps,
#         guidance_scale=req.guidance_scale,
#         start_step=1000,
#         conditioning_data=req.prompts,
#     )
#     images_b64 = []
#     for img in samples:
#         buf = io.BytesIO()
#         img_uint8 = ((img + 1.0) * 127.5).astype("uint8")
#         Image.fromarray(img_uint8).save(buf, format="PNG")
#         images_b64.append(base64.b64encode(buf.getvalue()).decode())
#     return {"images": images_b64}

# # Entry point
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("inference:app", host="0.0.0.0", port=8000)

import os
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from flaxdiff.inference.pipeline import DiffusionInferencePipeline
import base64, io
from PIL import Image

# Force JAX to CPU (if you have GPU, adjust/remove)
os.environ["JAX_PLATFORMS"] = os.getenv("JAX_PLATFORMS", "cpu")

# Initialize diffusion pipeline from W&B registry
pipeline = DiffusionInferencePipeline.from_wandb_registry(
    modelname='diffusion-oxford_flowers102-res256',
    project='mlops-msml605-project',
    entity='umd-projects',
)

# Request schema
class GenerateRequest(BaseModel):
    prompts: List[str]
    num_samples: Optional[int] = None
    resolution: int = 256
    diffusion_steps: int = 100
    guidance_scale: float = 3.0

app = FastAPI()

# Inference endpoint
@app.post("/generate")
def generate(req: GenerateRequest):
    samples = pipeline.generate_samples(
        num_samples=req.num_samples or len(req.prompts),
        resolution=req.resolution,
        diffusion_steps=req.diffusion_steps,
        guidance_scale=req.guidance_scale,
        start_step=1000,
        conditioning_data=req.prompts,
    )
    images_b64 = []
    for img in samples:
        buf = io.BytesIO()
        img_uint8 = ((img + 1.0) * 127.5).astype("uint8")
        Image.fromarray(img_uint8).save(buf, format="PNG")
        images_b64.append(base64.b64encode(buf.getvalue()).decode())
    return {"images": images_b64}

# Entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_mashup:app", host="0.0.0.0", port=8000)

