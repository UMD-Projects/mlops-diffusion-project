import logging
import os
import hmac
import hashlib
import subprocess # For running the evaluation script
from typing import Dict, Any, Optional

from fastapi import FastAPI, Header, HTTPException, status, Request, BackgroundTasks
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv
import uvicorn

# Load environment variables from .env file if it exists
load_dotenv()

###############################################################################
#  Config
###############################################################################

# Secret used to verify the signature from W&B
# Ensure this matches the secret configured in the W&B webhook settings
WANDB_WEBHOOK_SECRET: Optional[str] = os.getenv("WANDB_WEBHOOK_SECRET")

# Path to your evaluation script
EVALUATION_SCRIPT_PATH: str = os.getenv("EVALUATION_SCRIPT_PATH", "evaluation_pipeline.py")

###############################################################################
#  Pydantic model — covers common variables W&B exposes in webhooks
###############################################################################

class WandbWebhookPayload(BaseModel):
    # Based on common webhook structures and the example in the original code
    event_type: str = Field(..., example="LINK_ARTIFACT") # Or ADD_ARTIFACT_ALIAS, etc.
    event_author: Optional[str] = None # Sometimes included
    project_name: Optional[str] = None
    entity_name: Optional[str] = None
    artifact_collection_name: Optional[str] = None # e.g., model registry name like 'resnet50'
    artifact_version: Optional[str] = None # e.g., wandb-artifact://...
    artifact_version_string: Optional[str] = None # e.g., entity/project/name:version_or_alias

    # Tell Pydantic to accept unknown / extra keys W&B might send
    model_config = ConfigDict(extra='allow')

###############################################################################
#  FastAPI setup
###############################################################################
app = FastAPI(
    title="W&B Webhook Receiver & Evaluation Trigger",
    version="0.2.0",
    description="Receives W&B webhooks, verifies signature, and triggers evaluation script in the background."
)
logger = logging.getLogger("wandb_webhook")
# Configure logging format for better readability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


###############################################################################
#  Helper Functions
###############################################################################

def verify_wandb_signature(secret: Optional[str], request_body: bytes, signature_header: Optional[str]) -> None:
    """
    Verifies the HMAC-SHA256 signature of the incoming W&B webhook request.
    Raises HTTPException if verification fails or is impossible.
    """
    if not secret:
        # Log a warning but don't immediately fail if secret isn't configured during testing.
        # Verification will naturally fail if signature_header is present.
        logger.warning("WANDB_WEBHOOK_SECRET not configured. Signature verification skipped/will fail.")
        # If a signature was sent anyway, it's an error
        if signature_header:
             raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Webhook secret not configured on server.",
            )
        return # Allow request if no secret AND no signature header

    if not signature_header:
        logger.error("Signature header (X-Wandb-Signature) missing from webhook request.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing X-Wandb-Signature header",
        )

    try:
        # Format is 'sha256=<hex_digest>'
        method, signature_hash = signature_header.split('=', 1)
        if method != 'sha256':
            raise ValueError("Unsupported signature method.")

        # Calculate the expected signature
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            request_body,
            hashlib.sha256
        ).hexdigest()

        # Compare using a timing-safe method
        if not hmac.compare_digest(expected_signature, signature_hash):
            logger.error("Webhook signature mismatch.")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid webhook signature",
            )
        logger.info("Webhook signature verified successfully.")

    except ValueError as e:
        logger.error(f"Error parsing signature header: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid signature header format: {e}",
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during signature verification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error during signature verification.",
        )

def verify_token(expected_token: Optional[str], auth_header: Optional[str]) -> None:
    """Abort with 401 if bearer token is missing or wrong."""
    if auth_header is None or not auth_header.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")

    supplied = auth_header.split(" ", 1)[1]
    if supplied != expected_token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid bearer token")

def run_evaluation_job(registry_name: str, alias: str) -> None:
    """
    Runs the evaluation script in a separate process.
    Logs the outcome (success or failure to launch).
    """
    command = [
        "python3", # Or just "python" depending on your environment
        EVALUATION_SCRIPT_PATH,
        "--model_registry", registry_name,
        "--version", alias
    ]
    logger.info(f"Launching background evaluation job with command: {' '.join(command)}")
    try:
        # Use Popen for non-blocking execution.
        # Capture stdout/stderr if needed for more detailed logging, but keep it simple here.
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # We don't wait for completion here as it's a background task
        logger.info(f"Successfully launched evaluation process (PID: {process.pid}) for {registry_name}:{alias}")
        # Optional: You could store the PID or process object if you need to monitor it later
    except FileNotFoundError:
        logger.error(f"Evaluation script not found at '{EVALUATION_SCRIPT_PATH}'. Cannot launch job.")
    except Exception as e:
        logger.error(f"Failed to launch evaluation process for {registry_name}:{alias}. Error: {e}")


def handle_event(payload: WandbWebhookPayload, background_tasks: BackgroundTasks) -> None:
    """
    Processes the validated webhook payload and triggers background tasks.
    """
    event_type = payload.event_type
    version_string = payload.artifact_version_string or "N/A"

    logger.info(f"Handling event '{event_type}' for artifact version string: '{version_string}'")

    # Check for events indicating a model alias was added or changed
    # W&B might use different event types, adjust as needed based on webhook logs
    # Common ones could be ADD_ARTIFACT_ALIAS, LINK_ARTIFACT, artifact_promoted
    if event_type in ["LINK_ARTIFACT", "ADD_ARTIFACT_ALIAS"] and version_string != "N/A":
        try:
            # Extract registry name and alias (e.g., "resnet50:production")
            parts = version_string.split('/') # e.g., ["myteam", "model-registry", "resnet50:production"]
            if len(parts) >= 3:
                registry_and_alias = parts[-1] # "resnet50:production"
                registry_name, alias = registry_and_alias.rsplit(":", 1) # "resnet50", "production"

                logger.info(f"Triggering evaluation for alias '{alias}' on registry '{registry_name}'")
                # Add the evaluation job to run in the background
                background_tasks.add_task(run_evaluation_job, registry_name, alias)
            else:
                 logger.warning(f"Could not parse registry name and alias from artifact_version_string: {version_string}")

        except ValueError:
            logger.warning(f"Could not split registry name and alias in expected format from: {version_string}")
        except Exception as e:
             logger.error(f"Unexpected error processing event payload: {e}")

    # Handle other event types if needed
    elif event_type == "model_registry_version_created":
         logger.info(f"New model registry version created: {version_string}. Add specific handling if needed.")
         # Example: Maybe trigger a different kind of validation?
         # background_tasks.add_task(run_validation, payload.artifact_collection_name, payload.version_string)

    else:
        logger.info(f"No specific action configured for event type: {event_type}")


###############################################################################
#  HTTP Routes
###############################################################################

@app.post(
    "/evaluate",
    status_code=status.HTTP_202_ACCEPTED, # Use 202 Accepted for background tasks
    summary="Receive W&B Webhook and Trigger Evaluation",
    response_description="Acknowledged webhook receipt, evaluation triggered in background."
)
async def wandb_webhook_endpoint(
    request: Request,
    background_tasks: BackgroundTasks, # Inject background tasks handler
    x_wandb_signature: Optional[str] = Header(None, alias="X-Wandb-Signature"), # W&B Signature Header
    authorization: Optional[str] = Header(None, convert_underscores=False),
):
    """
    Receives webhook notifications from Weights & Biases, verifies the signature,
    and triggers a model evaluation script in the background if the event matches
    criteria (e.g., linking an artifact alias like 'production').

    Requires `X-Wandb-Signature` header for verification using `WANDB_WEBHOOK_SECRET`.
    """
    # 1. Get raw body for signature verification *before* parsing JSON
    raw_body = await request.body()

    # 2. Verify signature
    # verify_wandb_signature(WANDB_WEBHOOK_SECRET, raw_body, x_wandb_signature)
    verify_token(WANDB_WEBHOOK_SECRET, authorization)

    # 3. Parse JSON payload (safe now that signature is verified)
    try:
        body: Dict[str, Any] = await request.json()
        payload = WandbWebhookPayload(**body)  # validates & normalises using Pydantic
        logger.info(f"Parsed payload for event: {payload.event_type}")
    except Exception as e:
        logger.error(f"Failed to parse webhook JSON payload: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON payload: {e}",
        )

    # 4. Handle the event logic and schedule background tasks
    handle_event(payload, background_tasks)

    # 5. Return immediately while tasks run in background
    return {"status": "accepted", "message": "Webhook received and evaluation job scheduled."}


@app.get("/", summary="Health Check")
def read_root():
    """Basic health check endpoint."""
    return {"status": "healthy", "message": "W&B Webhook Receiver is running."}


###############################################################################
#  Entrypoint for running the server directly
###############################################################################
if __name__ == "__main__":
    if not WANDB_WEBHOOK_SECRET:
        logger.warning("WANDB_WEBHOOK_SECRET environment variable not set. "
                       "Webhook signature verification will be skipped or fail if signature is sent.")
    if not os.path.exists(EVALUATION_SCRIPT_PATH):
         logger.warning(f"Evaluation script path '{EVALUATION_SCRIPT_PATH}' not found. Background jobs will fail.")

    # Run the server using uvicorn
    # Use reload=True only for development
    port = int(os.getenv("PORT", 8000)) # Allow port configuration via env var
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=False) # Set reload=True for dev