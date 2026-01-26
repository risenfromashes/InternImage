#!/usr/bin/env bash
set -euo pipefail

# Load env vars
source .env

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
FULL_IMAGE_NAME="${ECR_URI}/${ECR_REPOSITORY}:${IMAGE_TAG}"

echo "🔐 Logging into ECR..."
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR_URI"

echo "🐳 Building Docker image..."
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo "🏷️  Tagging image..."
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "$FULL_IMAGE_NAME"

echo "🚀 Pushing image to ECR..."
docker push "$FULL_IMAGE_NAME"

echo "✅ Done!"
