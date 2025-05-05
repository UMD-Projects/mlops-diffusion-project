#!/bin/bash

# Check if API key is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <your_wandb_api_key>"
  exit 1
fi

# --- Configuration ---
export PROJECT_ID=$(gcloud config get-value project)
export REGION=europe-west4 # Or your desired region
export JOB_NAME="flaxdiff_wandb_sweep_$(date +%Y%m%d_%H%M%S)"
export IMAGE_URI="europe-west4-docker.pkg.dev/${PROJECT_ID}/flaxdiff-docker-repo/flaxdiff-tpu-trainer:latest" # Verify this URI
export SWEEP_ID="umd-projects/mlops-msml605-project/3s98m11b" # Your sweep ID
export WANDB_API_KEY=$1

# TPU Configuration
export ACCELERATOR_TYPE="TPU_V3"
export ACCELERATOR_COUNT=8

# Optional Service Account
# export SERVICE_ACCOUNT="your-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Define template and temporary file names
CONFIG_TEMPLATE="config_template.yaml" # Use the CORRECTLY STRUCTURED template
TEMP_YAML="temp_config_$(date +%s).yaml"
TEMP_JSON="temp_config_$(date +%s).json" # Output JSON file

# Check if template file exists
if [ ! -f "$CONFIG_TEMPLATE" ]; then
    echo "Error: Configuration template file '$CONFIG_TEMPLATE' not found."
    exit 1
fi

# Create a temporary YAML config file from the template
cp "$CONFIG_TEMPLATE" "$TEMP_YAML"

# Replace placeholders in the temporary YAML config file
sed -i "s|JOB_NAME_PLACEHOLDER|$JOB_NAME|g" "$TEMP_YAML"
sed -i "s|ACCELERATOR_TYPE_PLACEHOLDER|$ACCELERATOR_TYPE|g" "$TEMP_YAML"
sed -i "s|ACCELERATOR_COUNT_PLACEHOLDER|$ACCELERATOR_COUNT|g" "$TEMP_YAML"
sed -i "s|IMAGE_URI_PLACEHOLDER|$IMAGE_URI|g" "$TEMP_YAML"
sed -i "s|SWEEP_ID_PLACEHOLDER|$SWEEP_ID|g" "$TEMP_YAML"
sed -i "s|WANDB_API_KEY_PLACEHOLDER|$WANDB_API_KEY|g" "$TEMP_YAML"

# Optional service account substitution
# if [ -n "$SERVICE_ACCOUNT" ]; then
#   sed -i "s|# serviceAccount: SERVICE_ACCOUNT_PLACEHOLDER|serviceAccount: $SERVICE_ACCOUNT|g" "$TEMP_YAML"
# fi

echo "--- Generated YAML Config ($TEMP_YAML): ---"
cat "$TEMP_YAML"
echo "-------------------------------------------"

# --- Convert YAML to JSON ---
# Check for Python + PyYAML first, fallback to yq
if python -c "import sys, yaml, json" &> /dev/null; then
    echo "Converting YAML to JSON using Python..."
    # Use python -c to load YAML and dump JSON
    python -c 'import sys, yaml, json; json.dump(yaml.safe_load(open(sys.argv[1])), sys.stdout, indent=2)' "$TEMP_YAML" > "$TEMP_JSON"
    CONVERSION_STATUS=$?
elif command -v yq &> /dev/null; then
    echo "Converting YAML to JSON using yq..."
    yq eval -o=json "$TEMP_YAML" > "$TEMP_JSON"
    CONVERSION_STATUS=$?
else
    echo "Error: Need Python with PyYAML ('pip install PyYAML') or 'yq' ('apt install yq') to convert config."
    rm "$TEMP_YAML"
    exit 1
fi

# Check if conversion was successful
if [ $CONVERSION_STATUS -ne 0 ]; then
    echo "Error converting YAML to JSON."
    rm "$TEMP_YAML"
    # TEMP_JSON might be empty or partial, remove it too
    rm -f "$TEMP_JSON"
    exit 1
fi

echo "--- Generated JSON Config ($TEMP_JSON): ---"
cat "$TEMP_JSON"
echo "-------------------------------------------"

# --- Launch Command using custom-jobs with JSON config ---
gcloud ai custom-jobs create \
  --project=$PROJECT_ID \
  --region=$REGION \
  --display-name="$JOB_NAME" --config="$TEMP_JSON"

# Check exit status of gcloud command
if [ $? -eq 0 ]; then
  echo "Successfully submitted Vertex AI Custom Job: $JOB_NAME using config file $TEMP_JSON."
  # Clean up temporary files on success
  rm "$TEMP_YAML" "$TEMP_JSON"
else
  echo "Error submitting Vertex AI Custom Job: $JOB_NAME. Check output above."
  # Keep files for debugging
  exit 1
fi