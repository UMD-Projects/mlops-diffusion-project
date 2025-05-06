import streamlit as st
import requests
import time
from PIL import Image
from io import BytesIO
import base64
from dotenv import load_dotenv
import os

API_BASE = os.getenv("API_BASE", "http://diffusion-service:80")

st.title("🧠 Diffusion Inference Streamlit")

prompt = st.text_input("Enter your prompt:")
model_name = st.text_input("Model name (optional):", value="diffusion-laiona_coco-res256-sweep-wgjsicqf")
generate = st.button("Generate Image")

if generate and prompt:
    with st.spinner("Submitting job..."):
        response = requests.post(
            f"{API_BASE}/generate",
            json={"prompts": [prompt], "model_name": model_name}
        )
        if response.status_code == 200:
            job_id = response.json().get("job_id")
            st.success(f"Job submitted! Job ID: `{job_id}`")

            # Poll for result
            with st.spinner("Waiting for result (up to 8 minutes)..."):
                progress = st.progress(0)
                success = False
                total_attempts = 240  # 240 x 2s = 8 minutes

                for i in range(total_attempts):
                    result = requests.get(f"{API_BASE}/result/{job_id}?t={time.time()}")  # cache busting
                    data = result.json()

                    if data.get("status") == "completed":
                        images_b64 = data.get("result", [])
                        if isinstance(images_b64, list) and images_b64:
                            for idx, img_str in enumerate(images_b64):
                                img_data = base64.b64decode(img_str)
                                image = Image.open(BytesIO(img_data))
                                st.image(image, caption=f"Generated Image {idx + 1}")
                            success = True
                        else:
                            st.warning("No image returned.")
                            success = True
                        break

                    elif data.get("status") == "running":
                        progress.progress((i + 1) / total_attempts)
                        time.sleep(2)
                    else:
                        st.warning("Unexpected status or error.")
                        success = True
                        break

                if not success:
                    st.warning("Job is still running. Try again later.")
                    st.info(f"You can recheck the job status later using Job ID: `{job_id}`")
        else:
            st.error("Failed to submit job.")
