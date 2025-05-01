import os
os.environ["JAX_PLATFORMS"] = "cpu"

# inference.py


from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from flaxdiff.inference.pipeline import DiffusionInferencePipeline
import base64, io
from PIL import Image
import wandb


# Force JAX to CPU (if you have GPU, adjust/remove)
os.environ["JAX_PLATFORMS"] = os.getenv("JAX_PLATFORMS", "cpu")

app = FastAPI()

# Initialize diffusion pipeline from W&B registry

CURRENT_PIPELINE = None
DEFAULT_MODEL = "diffusion-oxford_flowers102-res256"
WAND_PROJECT = "mlops-msml605-project"
WAND_ENTITY = "umd-projects"


#load pipeline
def load_pipeline(model_name: str):
    global CURRENT_PIPELINE
    CURRENT_PIPELINE = DiffusionInferencePipeline.from_wandb_registry(
        modelname=model_name,
        project="mlops-msml605-project",
        entity="umd-projects",
    )
    return f"Model '{model_name}' loaded successfully."



#autoload default model onnstartup
@app.on_event("startup")
def startup_event():
    print(f"Loading default model: {DEFAULT_MODEL}")
    try:
        load_pipeline(DEFAULT_MODEL)
        print("Default model loaded.")
    except Exception as e:
        print(f"Model loading failed: {e}")



# Request schema
class GenerateRequest(BaseModel):
    prompts: List[str]
    num_samples: Optional[int] = None
    resolution: int = 256
    diffusion_steps: int = 100
    guidance_scale: float = 3.0





@app.post("/switch_model")
def switch_model(model_name: str):
    try:
        msg = load_pipeline(model_name)
        return {"status": "success", "message": msg}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#list available models from wandb registry
@app.get("/models")
def list_models():
    try:
        api = wandb.Api()
        registry = api.model_registry(WAND_ENTITY)
        models = [model.name for model in registry if model.name.startswith("diffusion-")]
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")


# Inference endpoint
@app.post("/generate")
def generate(req: GenerateRequest):
    if CURRENT_PIPELINE is None:
        raise HTTPException(status_code=400, detail="No model is currently loaded. Use /switch_model first.")
    
    run = wandb.init(project=WAND_PROJECT, job_type="inference", entity=WAND_ENTITY)
    run.config.update(req.dict())

    samples = CURRENT_PIPELINE.generate_samples(
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
        pil_img = Image.fromarray(img_uint8)
        pil_img.save(buf, format="PNG")
        images_b64.append(base64.b64encode(buf.getvalue()).decode())
        #log to wandb
        run.log({f"generated_image_{i}": wandb.Image(pil_img)})
    run.finish
    return {"images": images_b64}


# Entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_mashup:app", host="0.0.0.0", port=8000)

