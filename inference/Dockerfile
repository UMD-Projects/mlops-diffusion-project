FROM python:3.11

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*



WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir wandb pydantic flaxdiff transformers einops diffusers torch
RUN pip install --no-cache-dir jax[tpu]==0.5.3 flax[all] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install --no-cache-dir fastapi uvicorn optax pillow matplotlib
# # Copy requirements.txt first (important for caching)
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

#COPY inference_mashup.py .

# Expose port 8000 (the port used by Uvicorn).
EXPOSE 8000

# Command to run the server using Uvicorn.


CMD ["uvicorn", "inference_mashup:app", "--host", "0.0.0.0", "--port", "8000"]

