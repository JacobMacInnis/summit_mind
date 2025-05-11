#!/bin/bash

set -e

INPUT_MODEL=$1
VALID_MODELS=("small" "base")

PROJECT_ID="summit-mind"
REGION="us-east1"
REPO_NAME="summit-mind-repo" 
IMAGE_NAME="summit-mind-api"
# FULL_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest"
BASE_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME"

# Validate input
if [[ -n "$INPUT_MODEL" && ! " ${VALID_MODELS[@]} " =~ " ${INPUT_MODEL} " ]]; then
  echo "❌ ERROR: Unknown model '$INPUT_MODEL'. Use: 'small', 'base', or leave blank to deploy both."
  exit 1
fi

# Make sure you're logged into GCP
# gcloud auth login

# Set project
gcloud config set project $PROJECT_ID

# Configure docker to use gcloud credential helper
gcloud auth configure-docker $REGION-docker.pkg.dev

# Move to the script's directory
cd "$(dirname "$0")"
echo "📂 Working directory: $(pwd)"

# Confirm Dockerfile presence
echo "🔍 Dockerfile:"
ls -l Dockerfile

# Remove old buildx builder if it exists
echo "Clean up previous buildx builder (if exists)..."
docker buildx rm mybuilder 2>/dev/null || true


# Create new buildx builder
echo "Creat new buildx builder..."
docker buildx create --name mybuilder --use
docker buildx inspect mybuilder --bootstrap

# Decide which models to build
MODELS=("${VALID_MODELS[@]}")
if [[ -n "$INPUT_MODEL" ]]; then
  MODELS=("$INPUT_MODEL")
fi


for MODEL_NAME in "${MODELS[@]}"; do
  if [[ "$MODEL_NAME" == "small" ]]; then
    ACTUAL_MODEL_DIR="summit-mind-t5-small"
  elif [[ "$MODEL_NAME" == "base" ]]; then
    ACTUAL_MODEL_DIR="summit-mind-t5-base-final-pytorch"
  else
    echo "❌ ERROR: Unknown model '$MODEL_NAME'"
    exit 1
  fi
  TAGGED_IMAGE="$BASE_IMAGE-$MODEL_NAME:latest"

  echo "🐳 Building Docker image for model: $MODEL_NAME (path: $ACTUAL_MODEL_DIR)"
  docker buildx build --no-cache \
    --platform linux/amd64 \
    --build-arg MODEL_NAME=$ACTUAL_MODEL_DIR \
    --load \
    -t $IMAGE_NAME \
    -f Dockerfile .

  echo "🏷️ Tagging image as: $TAGGED_IMAGE"
  docker tag $IMAGE_NAME $TAGGED_IMAGE

  echo "📤 Pushing image: $TAGGED_IMAGE"
  docker push $TAGGED_IMAGE

   # Clean up local tag to avoid clutter
  docker rmi $IMAGE_NAME || true

  echo "🚀 Deploying to Cloud Run as: $IMAGE_NAME-$MODEL_NAME"
  gcloud run deploy $IMAGE_NAME-$MODEL_NAME \
    --image $TAGGED_IMAGE \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --cpu 4 \
    --memory 8Gi \
    --timeout 900 \
    --max-instances=2 \
    --set-env-vars MODEL_NAME=$MODEL_NAME


  echo "✅ Deployment complete for: $MODEL_NAME"
done

echo "🧹 Cleaning up local Docker images and build cache..."
docker image prune -f
docker builder prune -f
