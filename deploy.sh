#!/bin/bash

PROJECT_ID="summit-mind"
REGION="us-east1"
REPO_NAME="summit-mind-repo" 
IMAGE_NAME="summit-mind-api"

# Make sure you're logged into GCP
gcloud auth login

# Set project
gcloud config set project $PROJECT_ID

# Configure docker to use gcloud credential helper
gcloud auth configure-docker $REGION-docker.pkg.dev

# Build docker image
docker buildx build --no-cache --platform linux/amd64 -t $IMAGE_NAME .



# Tag docker image for GCP Artifact Registry
docker tag $IMAGE_NAME $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest

# Push docker image to Artifact Registry
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest
docker push us-east1-docker.pkg.dev/summit-mind/summit-mind-repo/summit-mind-api:latest
# Deploy to Cloud Run
gcloud run deploy $IMAGE_NAME \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --cpu 2 \
  --memory 4Gi \
  --timeout 900


echo "✅ Deployment complete! View your service URL above."

# gcloud run deploy summit-mind-api \
#   --image us-east1-docker.pkg.dev/summit-mind/summit-mind-repo/summit-mind-api:latest \
#   --platform managed \
#   --region us-east1 \
#   --allow-unauthenticated \
#   --cpu 2 \
#   --memory 4Gi \
#   --timeout 900