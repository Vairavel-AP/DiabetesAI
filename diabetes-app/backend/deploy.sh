#!/bin/bash
# ============================================================
# Diabetes Prediction API — Deploy Script
# Run this from the /diabetes-app/backend/ folder
# Requires: AWS CLI + SAM CLI installed & configured
# ============================================================

set -e

STACK_NAME="diabetes-prediction-api"
REGION="ap-south-1"
S3_BUCKET="diabetes-deploy-$(date +%s)"   # temp bucket for artifacts

echo "🚀 Creating S3 deployment bucket: $S3_BUCKET"
aws s3 mb s3://$S3_BUCKET --region $REGION

echo "📦 Building SAM application..."
sam build

echo "📤 Packaging & deploying to AWS..."
sam deploy \
  --stack-name $STACK_NAME \
  --s3-bucket $S3_BUCKET \
  --region $REGION \
  --capabilities CAPABILITY_IAM \
  --no-confirm-changeset

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Your API URL:"
aws cloudformation describe-stacks \
  --stack-name $STACK_NAME \
  --region $REGION \
  --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" \
  --output text

echo ""
echo "👆 Copy this URL into index.html → const API_URL = '...'"
