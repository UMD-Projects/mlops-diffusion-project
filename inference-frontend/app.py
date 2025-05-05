import streamlit as st
import requests
import time
from PIL import Image
from io import BytesIO
import base64

API_BASE = "http://34.122.168.182"

st.title("🧠 Diffusion Inference Streamlit")

prompt = st.text_input("Enter your prompt:")
model_name = st.text_input("Model name (optional):", value="diffusion-oxford_flowers102-res256")
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
                for _ in range(30):
                    result = requests.get(f"{API_BASE}/result/{job_id}")
                    data = result.json()

                    if data.get("status") == "completed":
                        images_b64 = data.get("result", [])
                        if images_b64:
                            for i, image_b64 in enumerate(images_b64):
                                img_data = base64.b64decode(image_b64)
                                image = Image.open(BytesIO(img_data))
                                st.image(image, caption=f"Generated Image {i+1}")
                        else:
                            st.error("No image data returned.")
                        break
                    elif data.get("status") == "running":
                        time.sleep(2)
                    else:
                        st.warning("Unexpected status or error.")
                        break
                else:
                    st.warning("Job is still running. Try again later.")
        else:
            st.error("Failed to submit job.")
