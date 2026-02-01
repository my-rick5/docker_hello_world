#!/bin/bash

# 1. Variables - Update these once
PROJECT_ID="dockerhelloworld-485819"
IMAGE_NAME="hello-world-app"
DEPLOYMENT_NAME="hello-world-deployment"
# This grabs the container name automatically so we don't guess again
CONTAINER_NAME=$(kubectl get deployment $DEPLOYMENT_NAME -o jsonpath='{.spec.template.spec.containers[0].name}')

# 2. Generate a unique tag using the current timestamp (e.g., v20240520-1430)
TAG="v$(date +%Y%m%d-%H%M%S)"
FULL_IMAGE_PATH="gcr.io/$PROJECT_ID/$IMAGE_NAME:$TAG"

echo "🚀 Starting deployment for $TAG..."

# 3. Build (AMD64 for GKE)
echo "📦 Building image..."
docker build -t $FULL_IMAGE_PATH --platform linux/amd64 .

# 4. Push
echo "☁️ Pushing to GCR..."
docker push $FULL_IMAGE_PATH

# 5. Update GKE
echo "☸️ Updating GKE deployment..."
kubectl set image deployment/$DEPLOYMENT_NAME $CONTAINER_NAME=$FULL_IMAGE_PATH

echo "✅ Success! Deployment $TAG is rolling out."
echo "Check files with: kubectl exec deployment/$DEPLOYMENT_NAME -- ls /app"