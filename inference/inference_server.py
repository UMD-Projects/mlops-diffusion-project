import uuid
import threading
import base64
import io
import numpy as np
from datetime import datetime

import jax
import jax.numpy as jnp
import optax
from PIL import Image

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

from flaxdiff.schedulers import EDMNoiseScheduler, KarrasVENoiseScheduler
from flaxdiff.predictors import KarrasPredictionTransform
from flaxdiff.models.simple_unet import Unet
from flaxdiff.trainer import DiffusionTrainer
from flaxdiff.utils import defaultTextEncodeModel
from flaxdiff.samplers.euler import EulerAncestralSampler

# --- Global Configurations ---
IMAGE_SIZE = 128

# --- Inference request schema ---
class InferenceRequest(BaseModel):
    prompt: str
    modelId: str = None

# --- Job status schema (for responses) ---
class JobResponse(BaseModel):
    job_id: str
    status: str
    result: str = None   # base64-encoded image when done
    error: str = None


jobs = {} 
jobs_lock = threading.Lock()


# --- Utility Functions ---
def jax_array_to_base64(sample: jnp.ndarray) -> str:
    """
    Convert a JAX array (with shape [H, W, 3]) in normalized range [-1, 1]
    to a PNG image encoded in base64.
    """
    # Convert to NumPy, denormalize and clip to [0, 255]
    np_img = np.array(sample)
    np_img = np.clip((np_img + 1.0) * 127.5, 0, 255).astype(np.uint8)
    im = Image.fromarray(np_img)
    buffered = io.BytesIO()
    im.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# --- Load Model on Startup ---
def load_model():
    """
    Initialize and load the diffusion model, text encoder and sampler.
    Adjust checkpoint_path or any other parameters to match your setup.
    """
    # Load text encoder
    text_encoder = defaultTextEncodeModel()

    # Define noise schedulers
    edm_schedule = EDMNoiseScheduler(1, sigma_max=80, rho=7, sigma_data=0.5)
    karas_ve_schedule = KarrasVENoiseScheduler(1, sigma_max=80, rho=7, sigma_data=0.5)

    # Define input shapes
    input_shapes = {
        "x": (IMAGE_SIZE, IMAGE_SIZE, 3),
        "temb": (),
        "textcontext": (77, 768)
    }

    # Define the UNet model architecture
    unet = Unet(
        emb_features=256,
        feature_depths=[64, 64, 128, 256, 512],
        attention_configs=[
            None,
            {"heads": 8, "dtype": jnp.float32, "flash_attention": False, "use_projection": False, "use_self_and_cross": True},
            {"heads": 8, "dtype": jnp.float32, "flash_attention": False, "use_projection": False, "use_self_and_cross": True},
            {"heads": 8, "dtype": jnp.float32, "flash_attention": False, "use_projection": False, "use_self_and_cross": True},
            {"heads": 8, "dtype": jnp.float32, "flash_attention": False, "use_projection": False, "use_self_and_cross": False}
        ],
        num_res_blocks=2,
        num_middle_res_blocks=1
    )

    # Define optimizer
    solver = optax.adam(2e-4)

    name = "prototype-edm-Diffusion_SDE_VE_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    checkpoint_path = '/home/mrwhite0racle/mlops-diffusion-project/prototype-checkpoint/diffusion_sde_ve_2025-04-08_16:00:25'

    # Create trainer instance
    trainer = DiffusionTrainer(
        unet,
        optimizer=solver,
        input_shapes=input_shapes,
        noise_schedule=edm_schedule,
        rngs=jax.random.PRNGKey(4),
        name=name,
        model_output_transform=KarrasPredictionTransform(sigma_data=edm_schedule.sigma_data),
        encoder=text_encoder,
        distributed_training=True,
        wandb_config=None,
        load_from_checkpoint=checkpoint_path
    )

    # Prepare null labels for the sampler
    null_labels_full = text_encoder([""])

    # Create sampler instance using Euler ancestral sampling
    sampler = EulerAncestralSampler(
        trainer.model,
        None,
        noise_schedule=karas_ve_schedule,
        image_size=IMAGE_SIZE,
        autoencoder=trainer.autoencoder,
        model_output_transform=trainer.model_output_transform,
        guidance_scale=4,
        null_labels_seq=null_labels_full
    )

    return {
        "trainer": trainer,
        "text_encoder": text_encoder,
        "sampler": sampler
    }


# Load the model once and keep it in memory.
model_data = load_model()


# --- Background Inference Task ---
def process_job(job_id: str, prompt: str):
    """
    Run the diffusion model inference for a given prompt,
    update the job result once the image is generated.
    """
    try:
        with jobs_lock:
            jobs[job_id]["status"] = "in_progress"
        trainer = model_data["trainer"]
        sampler = model_data["sampler"]
        text_encoder = model_data["text_encoder"]

        # Get the EMA parameters from the trainer state
        ema_params = trainer.get_state().ema_params

        # Generate a single image based on the prompt. 
        # (Assumes your sampler accepts a list of prompts.)
        samples = sampler.generate_images(
            params=ema_params,
            num_images=1,
            diffusion_steps=200,
            start_step=1000,
            end_step=0,
            priors=None,
            model_conditioning_inputs=(text_encoder([prompt]),)
        )
        # Extract the generated image from the returned batch.
        sample = samples[0]

        # Convert the generated image to base64.
        result_img = jax_array_to_base64(sample)
        with jobs_lock:
            jobs[job_id]["status"] = "complete"
            jobs[job_id]["result"] = result_img
    except Exception as e:
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)


# --- Create FastAPI app ---
app = FastAPI(title="Diffusion Inference API")


@app.get("/v1/infer", response_model=dict)
def list_running_jobs():
    """
    Return a dictionary of job IDs for jobs that are still running (pending or in progress).
    """
    with jobs_lock:
        running_jobs = {jid: job for jid, job in jobs.items() if job["status"] != "complete"}
    return running_jobs


@app.post("/v1/infer", response_model=JobResponse)
def submit_inference(req: InferenceRequest, background_tasks: BackgroundTasks):
    """
    Submit a new inference job.
    The request must contain a prompt and optionally a modelId.
    Returns a unique job ID.
    """
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {"status": "pending", "result": None, "error": None, "prompt": req.prompt}
    # Schedule the background job
    background_tasks.add_task(process_job, job_id, req.prompt)
    return JobResponse(job_id=job_id, status="pending")


@app.get("/v1/infer/{job_id}", response_model=JobResponse)
def get_job_status(job_id: str):
    """
    Get the status of the given job ID.
    If the job is complete, the base64-encoded image is returned as well.
    """
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        job = jobs[job_id]
    return JobResponse(job_id=job_id, status=job["status"], result=job.get("result"), error=job.get("error"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("inference_server:app", host="0.0.0.0", port=8000, reload=True)
