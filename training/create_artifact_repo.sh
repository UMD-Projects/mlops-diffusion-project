gcloud artifacts repositories create flaxdiff-docker-repo \
    --repository-format=docker \
    --location=europe-west4 \
    --description="Docker repository for FlaxDiff training" 