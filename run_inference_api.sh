#!/bin/bash

# Step 1: Build Docker image
echo "🔧 Building Docker image..."
docker build -t diffusion-inference-api .

# Step 2: Run container (detached mode, port 8000:8000)
echo "🚀 Running container..."
docker run -d --name inference-container -p 8000:8000 diffusion-inference-api

# Step 3: Wait for container to start
echo "⏳ Waiting for the API to be ready..."
sleep 5

# Step 4: Test with a curl request
echo "📡 Sending inference request..."
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompts": ["a field of sunflowers under a blue sky"],
           "num_samples": 1,
           "resolution": 256,
           "diffusion_steps": 25,
           "guidance_scale": 3.0
         }'

echo -e "\n✅ Done!"

#eyJhbGciOiJSUzI1NiIsImtpZCI6ImdBcFVNREVCQnZqOEVlZmZGZVZtR09vcllicDJ0QTh1aGRHcEpjaVB6dEEifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJrdWJlcm5ldGVzLWRhc2hib2FyZCIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VjcmV0Lm5hbWUiOiJkYXNoYm9hcmQtYWRtaW4tc2EtdG9rZW4iLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC5uYW1lIjoiZGFzaGJvYXJkLWFkbWluLXNhIiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZXJ2aWNlLWFjY291bnQudWlkIjoiMWYzMTFhMzItNDJhMi00Yzg3LWJkN2MtZWU5MDdhMTc5YjRiIiwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50Omt1YmVybmV0ZXMtZGFzaGJvYXJkOmRhc2hib2FyZC1hZG1pbi1zYSJ9.e8gbX5-cQj6vyEpA5SDdhDbts-XkvMbHG3XybHghzoU06ejhOO_3PBAFq8Nxh0rcn8zQD0RGiH-BoldehhU8Bfgbm06McKJqm4SYst5TxUhl0R5N7Sra2Lu5Lx-6OvbhtgcotVJtcIgvwGlPDfmsEbAm8xgz9h_a5irtPu6CRwIRvr3jlT-vdkp0TE-PDQRO5fkMrd2cxBA_HgtZ7rXYYZG5xTM3-3Dt-tX38dkQCwB20RFrbBDabF0twcwr7dcqxsifs1kh-jBuC9yt8WT92HyaYLZOo2oW3EOJ7GPb1TtOjqxIsKfOV-jLuQHFmpNyYZnLOUfAAlsHrfj0WOU_MA% 