import uuid
import threading
import time
from app.model import model_data, jax_array_to_base64

# Job store: jobId -> {"status": "pending"/"running"/"complete"/"failed", "result": image_base64 or error}
jobs = {}

def create_job(prompt: str, model_id: str = None):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "result": None}
    thread = threading.Thread(target=run_job, args=(job_id, prompt, model_id))
    thread.start()
    return job_id

def run_job(job_id: str, prompt: str, model_id: str = None):
    from .inference import generate_image_base64
    jobs[job_id]["status"] = "running"
    try:
        img_base64 = generate_image_base64(prompt, model_id)
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["result"] = img_base64
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["result"] = str(e)

#mockjobs

# def run_job(job_id: str, prompt: str, model_id: str = None):
#     jobs[job_id]["status"] = "running"
#     try:
#         # Simulate processing time
#         time.sleep(2)

#         # Simulate result: a placeholder base64 image (1x1 transparent PNG)
#         fake_img = (
#             "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNg"
#             "YAAAAAMAAWgmWQ0AAAAASUVORK5CYII="
#         )
#         jobs[job_id]["status"] = "complete"
#         jobs[job_id]["result"] = fake_img
#     except Exception as e:
#         jobs[job_id]["status"] = "failed"
#         jobs[job_id]["result"] = str(e)


def get_job_status(job_id: str):
    return jobs.get(job_id, None)

def get_all_jobs():
    return {k: v["status"] for k, v in jobs.items()}

def process_job(job_id: str, prompt: str):
    try:
        with jobs_lock:
            jobs[job_id]["status"] = "in_progress"

        trainer = model_data["trainer"]
        sampler = model_data["sampler"]
        text_encoder = model_data["text_encoder"]

        encoded_prompt = text_encoder([prompt])
        samples = sampler.generate_images(
            params=trainer.get_state().ema_params,
            num_images=1,
            diffusion_steps=200,
            start_step=1000,
            end_step=0,
            priors=None,
            model_conditioning_inputs=(encoded_prompt,)
        )

        image_base64 = jax_array_to_base64(samples[0])

        with jobs_lock:
            jobs[job_id]["status"] = "complete"
            jobs[job_id]["result"] = image_base64

    except Exception as e:
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
