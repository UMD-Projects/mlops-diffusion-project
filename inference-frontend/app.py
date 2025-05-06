import streamlit as st
import requests
import time
from PIL import Image
from io import BytesIO
import base64

API_BASE = "http://localhost:8000"

st.title("🧠 Diffusion Inference Streamlit")

prompt = st.text_input("Enter your prompt:")
model_name = st.text_input("Model name (optional):", value="diffusion-oxford_flowers102-res128-sweep-d4es07fm")
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
            with st.spinner("Waiting for result..."):
                progress = st.progress(0)
                success = False

                for i in range(30):
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
                        progress.progress((i + 1) / 30)
                        time.sleep(2)
                    else:
                        st.warning("Unexpected status or error.")
                        success = True
                        break

                if not success:
                    st.warning("Job is still running. Try again later.")
        else:
            st.error("Failed to submit job.")
