### flower_diffusion_api/app/main.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.jobs import create_job, get_all_jobs, get_job_status, process_job
from app.model import load_model
import uuid 


class PromptRequest(BaseModel):
    prompt: str
    modelId: str | None = None

class JobResponse(BaseModel):
    job_id: str
    status: str
    result: str | None = None
    error: str | None = None


#Load Model
model_data = load_model()

app = FastAPI()

@app.get("/v1/infer")
def list_jobs():
    return {"jobs": get_all_jobs()}

@app.post("/v1/infer", response_model= JobResponse)
def start_inference(req: PromptRequest, background_tasks = BackgroundTasks):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    job_id = create_job(req.prompt, req.modelId)
    background_tasks.add_task(process_job, job_id, req.prompt)
    return JobResponse({"jobId": job_id, "status": "pending"})



@app.get("/v1/infer/{job_id}")
def get_job(job_id: str, response_model = JobResponse):
    job = get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(
        job_id=job_id,
        status=job["status"],
        result=job.get("result"),
        error=job.get("error")
    )




