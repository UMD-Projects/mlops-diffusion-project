# Use a base image with Python
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Prevent interactive prompts during installations
ENV DEBIAN_FRONTEND=noninteractive

# Install essential tools + libgl1 (needed for opencv/some graphics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bash \
    curl \
    git \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN mkdir -p /miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda3/miniconda.sh && \
    bash /miniconda3/miniconda.sh -b -u -p /miniconda3 && \
    rm /miniconda3/miniconda.sh
ENV PATH=/miniconda3/bin:$PATH

# Create Conda environment
RUN conda create -n flaxdiff python=3.12 -y

# Activate conda environment and install packages
# Note: This RUN command executes in a single shell layer to keep the conda env active
RUN echo "source activate flaxdiff" > ~/.bashrc && \
    . /miniconda3/bin/activate flaxdiff && \
    pip install --no-cache-dir jax[tpu]==0.5.3 flax[all] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
    pip install --no-cache-dir --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu && \
    pip install --no-cache-dir \
        tensorflow[cpu] \
        diffusers \
        orbax \
        optax \
        clu \
        grain \
        augmax \
        albumentations \
        datasets \
        transformers \
        opencv-python \
        pandas \
        tensorflow-datasets \
        jupyterlab \
        python-dotenv \
        scikit-learn \
        termcolor \
        wrapt \
        wandb \
        flaxdiff \
        gcsfs \
        decord \
        video-reader-rs \
        colorlog && \
    conda clean -afy

# Set TOKENIZERS_PARALLELISM env var (will be inherited by processes)
ENV TOKENIZERS_PARALLELISM=false

# Copy your training script into the container
COPY training.py /app/training.py

# (Optional) Set a default entrypoint - Vertex AI will override this
# ENTRYPOINT ["python", "/app/training.py"]
# For wandb sweeps, the command will be `wandb agent ...`

# Ensure the conda environment is activated for subsequent commands/entrypoint
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "flaxdiff"]
# CMD will be provided by Vertex AI job spec (e.g., ["wandb", "agent", "..."])