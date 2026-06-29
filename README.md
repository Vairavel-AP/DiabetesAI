# DiabetesAI
# 🩺 DiabetesAI — End-to-End AWS Machine Learning Pipeline

> A production-grade diabetes risk prediction system built entirely on AWS — from raw data ingestion to a live public web application powered by a real-time SageMaker inference endpoint.

![AWS](https://img.shields.io/badge/AWS-SageMaker%20Canvas-orange?logo=amazonaws)
![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Lambda](https://img.shields.io/badge/AWS-Lambda-yellow?logo=awslambda)
![API Gateway](https://img.shields.io/badge/AWS-API%20Gateway-purple?logo=amazonaws)
![S3](https://img.shields.io/badge/AWS-S3%20Hosted-red?logo=amazons3)
![Accuracy](https://img.shields.io/badge/Accuracy-84.4%25-green)
![AUC](https://img.shields.io/badge/AUC--ROC-0.898-brightgreen)
![Status](https://img.shields.io/badge/Status-Live%20%F0%9F%9F%A2-brightgreen)

---

## 🌐 Live Demo

| Resource         | URL                                                                  |
| ---------------- | -------------------------------------------------------------------- |
| **Frontend App** | http://diabetesai-frontend.s3-website.ap-south-1.amazonaws.com       |
| **API Endpoint** | https://3wnsbonc51.execute-api.ap-south-1.amazonaws.com/prod/predict |
| **AWS Region**   | ap-south-1 (Mumbai)                                                  |

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Complete ML Pipeline](#-complete-ml-pipeline)
- [AWS Architecture](#-aws-architecture)
- [AWS Tech Stack](#-aws-tech-stack)
- [Model Performance](#-model-performance)
- [Input Features](#-input-features)
- [Project Structure](#-project-structure)
- [Deployment Guide](#-deployment-guide)
- [API Reference](#-api-reference)
- [Frontend UI Features](#-frontend-ui-features)
- [Security Considerations](#-security-considerations)
- [Future Enhancements](#-future-enhancements)
- [Disclaimer](#️-disclaimer)

---

## 🎯 Project Overview

**DiabetesAI** is a complete, production-ready machine learning pipeline hosted on AWS that predicts diabetes risk from patient clinical data in real time. Built using exclusively AWS managed services, it demonstrates a full MLOps workflow — from raw data ingestion and preprocessing, through AutoML training and evaluation, to serverless deployment and a live interactive web dashboard hosted on S3.

### Key Highlights

- ✅ No-code ML pipeline using **Amazon SageMaker Canvas AutoML**
- ✅ Real-time inference via a secure **Lambda + API Gateway** proxy
- ✅ Interactive dark-themed **clinical dashboard UI** (HTML/CSS/JS)
- ✅ Infrastructure-as-Code deployment using **AWS SAM**
- ✅ Frontend hosted on **Amazon S3 Static Website** — publicly accessible
- ✅ **84.4% accuracy** with **0.898 AUC-ROC** on the Pima Indians Diabetes Dataset
- ✅ Full CORS-enabled REST API deployed in under 10 minutes
- ✅ **Fully live and accessible** to anyone worldwide

---

## 📊 Dataset Description

### Pima Indians Diabetes Dataset

Originally from the **National Institute of Diabetes and Digestive and Kidney Diseases**, this is one of the most widely studied datasets in medical machine learning research.

| Property       | Value                                           |
| -------------- | ----------------------------------------------- |
| Source         | Kaggle / UCI ML Repository                      |
| Total Rows     | 768 patient records                             |
| Total Columns  | 9 (8 features + 1 target)                       |
| Target Column  | `Outcome` — 0 (No Diabetes) / 1 (Diabetes)      |
| Class Split    | 500 Non-Diabetic (65.1%) / 268 Diabetic (34.9%) |
| Missing Values | Zeros in medical columns represent missing data |

### Data Quality Issues Identified & Fixed

| Column        | Issue                        | Fix Applied       | Fill Value |
| ------------- | ---------------------------- | ----------------- | ---------- |
| Glucose       | Zero = clinically impossible | Median imputation | 117        |
| BloodPressure | Zero = clinically impossible | Median imputation | 72         |
| SkinThickness | Zero = unmeasured            | Median imputation | 29         |
| Insulin       | Zero = unmeasured            | Median imputation | 125        |
| BMI           | Zero = clinically impossible | Median imputation | 32         |

> **Why Median?** These columns are right-skewed. Median is robust to the outlier zeros and gives a more accurate central value than mean.

---

## 🔄 Complete ML Pipeline

The pipeline follows industry-standard MLOps practices across **10 sequential stages**, each handled by a dedicated AWS service:

```
Stage 1  ──►  Stage 2  ──►  Stage 3  ──►  Stage 4  ──►  Stage 5
   S3          Data           Data          Feature        Canvas
 Ingestion   Wrangler EDA  Wrangler      Engineering      AutoML
                           Preprocess                     Training
                                                             │
Stage 10  ◄──  Stage 9  ◄──  Stage 8  ◄──  Stage 7  ◄──  Stage 6
   S3           Frontend      Lambda +      SageMaker      Canvas
Static Web     Dashboard     API GW        Endpoint      Evaluation
 Hosting
```

| #   | Phase                     | AWS Service               | Details                                            |
| --- | ------------------------- | ------------------------- | -------------------------------------------------- |
| 1   | Data Ingestion            | Amazon S3                 | diabetes.csv uploaded to S3 bucket (ap-south-1)    |
| 2   | Exploratory Data Analysis | SageMaker Data Wrangler   | Visual profiling, histograms, correlation analysis |
| 3   | Data Preprocessing        | SageMaker Data Wrangler   | Fill missing zeros with median values per column   |
| 4   | Feature Engineering       | SageMaker Data Wrangler   | Column selection, data type casting, flow export   |
| 5   | Model Training            | SageMaker Canvas AutoML   | Quick Build → Standard Build (extended HPO)        |
| 6   | Model Evaluation          | SageMaker Canvas          | Confusion matrix, AUC-ROC, F1, Recall analysis     |
| 7   | Model Deployment          | SageMaker Canvas Endpoint | Real-time REST inference endpoint (ap-south-1)     |
| 8   | API Layer                 | AWS Lambda + API Gateway  | Secure SageMaker proxy, deployed via AWS SAM       |
| 9   | Frontend                  | HTML / CSS / JS           | Dark clinical dashboard with sliders and gauge     |
| 10  | Hosting                   | Amazon S3 Static Website  | Publicly accessible, globally available            |

### Stage Details

#### Stage 1 — Data Ingestion (Amazon S3)

- `diabetes.csv` uploaded to S3 bucket in `ap-south-1`
- Folder structure: `raw/`, `processed/`, `model-artifacts/`
- S3 bucket serves as the single source of truth for the pipeline

#### Stage 2 — EDA (SageMaker Data Wrangler)

- Visual data profiling: histograms, box plots, correlation heatmap
- Data type detection: all 8 features confirmed as numeric (float/long)
- Identified zero-as-missing pattern across 5 clinical columns
- 768 rows × 9 columns confirmed, no structural nulls

#### Stage 3 — Preprocessing (SageMaker Data Wrangler)

- Transform: **Fill Missing** applied to BloodPressure, SkinThickness, Insulin, BMI, Glucose
- Strategy: **Median imputation** (robust to skew from zero outliers)
- Initial pipeline included PCA — **removed** after discovering model expected 7 PCA components instead of 8 raw features
- Final pipeline: raw 8-feature CSV → SageMaker endpoint (no PCA, no dimensionality reduction)

#### Stage 4 — Feature Engineering (SageMaker Data Wrangler)

- Column type casting (long/float) for SageMaker Canvas compatibility
- Target column `Outcome` set as label for binary classification
- Exported dataset registered directly to SageMaker Canvas

#### Stage 5 — Model Training (SageMaker Canvas AutoML)

- **Quick Build** (~15 min): rapid AutoML sweep across multiple algorithms → 81.17% accuracy
- **Standard Build** (~2 hrs): extended hyperparameter optimization → 84.4% accuracy
- Canvas internally tests XGBoost, Linear Learner, AutoGluon ensemble methods
- Best model automatically selected based on F1 Score (optimization metric for imbalanced data)

#### Stage 6 — Model Evaluation (SageMaker Canvas)

- Confusion matrix reviewed: True Positives, False Negatives analyzed
- AUC-ROC of **0.898** confirms very strong discriminative ability
- Recall prioritized: **85.18%** catches most actual diabetic patients
- Precision-Recall curve analyzed for optimal classification threshold

#### Stage 7 — Deployment (SageMaker Canvas Endpoint)

- One-click deployment from Canvas → Deploy tab
- **Endpoint:** `canvas-new-deployment-06-29-2026-12-09-AM`
- Region: `ap-south-1` (Mumbai)
- Accepts CSV payload, returns JSON with prediction + probability score

#### Stage 8 — API Layer (Lambda + API Gateway)

- Lambda (Python 3.13) acts as secure SageMaker proxy
- Handles AWS IAM signing — browser never needs AWS credentials
- API Gateway provides public HTTPS REST endpoint with full CORS headers
- Infrastructure deployed via **AWS SAM** (`template.yaml`) — one-command deploy

#### Stage 9 — Frontend Dashboard

- Dark clinical dashboard with 8 interactive range sliders
- Animated SVG donut gauge for diabetes probability visualization
- Color-coded verdict card, 3-metric panel, prediction history table
- Demo mode available before API URL is configured

#### Stage 10 — Hosting (Amazon S3 Static Website) ✅ LIVE

- `index.html` uploaded to S3 bucket `diabetesai-frontend`
- Block Public Access disabled, public bucket policy applied
- Static website hosting enabled — no server required
- **Live URL:** http://diabetesai-frontend.s3-website.ap-south-1.amazonaws.com

---

## 🏗️ AWS Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                             │
│    http://diabetesai-frontend.s3-website.ap-south-1.amazonaws   │
│                        .com                                     │
│   8 Clinical Sliders → Click "Run Prediction"                   │
└───────────────┬─────────────────────────────────────────────────┘
                │
                │ Served from
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  AMAZON S3 STATIC WEBSITE                       │
│              Bucket: diabetesai-frontend                        │
│              index.html  ·  Public Read Policy                  │
└───────────────┬─────────────────────────────────────────────────┘
                │
                │ POST /predict (application/json)
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AMAZON API GATEWAY                           │
│    https://3wnsbonc51.execute-api.ap-south-1.amazonaws.com      │
│              REST API  ·  HTTPS  ·  CORS Enabled                │
└───────────────┬─────────────────────────────────────────────────┘
                │ Trigger
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AWS LAMBDA FUNCTION                         │
│           diabetes-prediction-api  ·  Python 3.13               │
│                                                                 │
│  1. Parse JSON body  (8 features)                               │
│  2. Format CSV: "6,148,72,35,125,33.6,0.627,50"                │
│  3. InvokeEndpoint with IAM auth (SigV4)                        │
│  4. Parse + return { prediction, probability, label }           │
└───────────────┬─────────────────────────────────────────────────┘
                │ InvokeEndpoint (IAM SigV4 signed)
                ▼
┌─────────────────────────────────────────────────────────────────┐
│              AMAZON SAGEMAKER CANVAS ENDPOINT                   │
│      canvas-new-deployment-06-29-2026-12-09-AM                  │
│                   Region: ap-south-1                            │
│                                                                 │
│  Model: AutoML Standard Build (Best Algorithm)                  │
│  Input:  8 raw clinical features (CSV)                          │
│  Output: { predicted_label: 1, probability: 0.78 }              │
└───────────────┬─────────────────────────────────────────────────┘
                │ JSON Response → Lambda → API Gateway → Browser
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BROWSER RENDERS RESULT                       │
│  • Animated SVG probability gauge                               │
│  • Color-coded verdict (Diabetic / Non-Diabetic)                │
│  • Confidence %, Risk Level, Prediction label                   │
│  • Row added to prediction history table                        │
└─────────────────────────────────────────────────────────────────┘

         Supporting AWS Services
         ────────────────────────────────────────
         Amazon S3            Dataset + artifacts
         SageMaker Canvas     AutoML train & eval
         Data Wrangler        EDA & preprocessing
         AWS CloudWatch       Lambda monitoring
         AWS IAM              Role-based access
         AWS SAM              Infrastructure as Code
```

### Request Flow — Step by Step

1. User visits the **S3-hosted frontend** and adjusts 8 clinical sliders
2. Clicks **Run Prediction** → browser sends `POST` JSON to API Gateway
3. **API Gateway** triggers Lambda function (Python 3.13)
4. **Lambda** validates input, formats CSV, calls SageMaker with IAM auth
5. **SageMaker Canvas** model runs inference on the 8-feature CSV
6. Lambda parses response: `{ prediction, probability, label }`
7. API Gateway returns CORS-enabled JSON to browser
8. Frontend renders animated gauge, verdict card, updates history table

---

## 🛠️ AWS Tech Stack

| AWS Service                 | Category           | Purpose                                          |
| --------------------------- | ------------------ | ------------------------------------------------ |
| **Amazon S3** (2 buckets)   | Storage + Hosting  | Dataset storage & static frontend hosting        |
| **SageMaker Data Wrangler** | Data Preprocessing | No-code EDA, median imputation, feature typing   |
| **SageMaker Canvas**        | AutoML Platform    | Model training, evaluation, one-click deployment |
| **AWS Lambda**              | Serverless Compute | Python 3.13 SageMaker proxy function             |
| **Amazon API Gateway**      | API Management     | REST API with CORS, public HTTPS endpoint        |
| **AWS SAM**                 | IaC / DevOps       | Infrastructure-as-Code — Lambda + API Gateway    |
| **AWS CloudWatch**          | Monitoring         | Lambda invocation logs, error tracking           |
| **AWS IAM**                 | Security           | Role-based `sagemaker:InvokeEndpoint` permission |

---

## 📈 Model Performance

### Quick Build vs Standard Build Comparison

| Metric        | Quick Build | Standard Build | Improvement |
| ------------- | ----------- | -------------- | ----------- |
| **Accuracy**  | 81.169%     | **84.416%**    | +3.25% ✅   |
| **F1 Score**  | 0.743       | **0.793**      | +0.05 ✅    |
| **Precision** | 71.186%     | **74.194%**    | +3.0% ✅    |
| **Recall**    | 77.778%     | **85.185%**    | +7.4% 🔥    |
| **AUC-ROC**   | 0.878       | **0.898**      | +0.02 ✅    |

### AUC-ROC Rating

| AUC-ROC Range   | Rating                            |
| --------------- | --------------------------------- |
| 0.90 – 1.00     | Outstanding                       |
| **0.80 – 0.90** | **Very Good ← This model: 0.898** |
| 0.70 – 0.80     | Good                              |
| 0.60 – 0.70     | Fair                              |

### Why Recall Is the Critical Metric

> In clinical diabetes screening, **missing a diabetic patient (False Negative) is far more dangerous** than a false alarm (False Positive). The Standard Build's **85.18% Recall** means the model correctly identifies 85 out of every 100 actual diabetic patients — a 7.4% improvement over Quick Build.

### Confusion Matrix — Standard Build

```
                     Predicted: Non-Diabetic    Predicted: Diabetic
                    ┌──────────────────────────┬────────────────────┐
Actual: Diabetic    │   False Negative (FN) ⚠️  │  True Positive ✅  │
                    │   Missed diabetic patient  │  Correctly caught  │
                    ├──────────────────────────┼────────────────────┤
Actual:             │   True Negative ✅         │  False Positive ℹ️ │
Non-Diabetic        │   Correctly cleared        │  Healthy flagged   │
                    └──────────────────────────┴────────────────────┘
```

---

## 🩺 Input Features

| Feature           | JSON Key        | Unit  | Description                 | Range       |
| ----------------- | --------------- | ----- | --------------------------- | ----------- |
| Pregnancies       | `pregnancies`   | count | Number of times pregnant    | 0 – 17      |
| Glucose           | `glucose`       | mg/dL | 2-hour plasma glucose (GTT) | 44 – 199    |
| Blood Pressure    | `bloodPressure` | mmHg  | Diastolic blood pressure    | 24 – 122    |
| Skin Thickness    | `skinThickness` | mm    | Triceps skinfold thickness  | 7 – 99      |
| Insulin           | `insulin`       | μU/mL | 2-hour serum insulin level  | 14 – 846    |
| BMI               | `bmi`           | kg/m² | Body mass index             | 18.0 – 67.0 |
| Diabetes Pedigree | `dpf`           | score | Genetic diabetes likelihood | 0.08 – 2.42 |
| Age               | `age`           | years | Patient age                 | 21 – 81     |

---

## 📁 Project Structure

```
DiabetesAI-Project/
├── frontend/
│   └── index.html                  ← Dark clinical dashboard UI (S3 hosted)
│
├── lambda/
│   └── lambda_function.py          ← AWS Lambda SageMaker proxy (Python 3.13)
│
├── backend/
│   ├── template.yaml               ← AWS SAM IaC (Lambda + API Gateway)
│   ├── lambda/
│   │   └── lambda_function.py      ← Lambda code copy for SAM build
│   └── deploy.sh                   ← One-command deploy script
│
└── README.md                       ← This file
```

---

## 🚀 Deployment Guide

### Prerequisites

```bash
# 1. Configure AWS CLI
aws configure
# Enter: Access Key ID, Secret Access Key
# Default region: ap-south-1
# Default output format: json

# 2. Install SAM CLI
pip install aws-sam-cli

# 3. Verify Python version
python --version   # Needs Python 3.13.x
```

### Step 1 — Prepare Lambda Code

```bash
cd DiabetesAI-Project/diabetes-app/backend
cp -r ../lambda ./lambda
```

### Step 2 — Deploy Backend (Lambda + API Gateway)

```bash
./deploy.sh
```

Expected output:

```
✅ Deployment complete!

Your API URL:
https://xxxxxxxxxx.execute-api.ap-south-1.amazonaws.com/prod/predict
```

### Step 3 — Update Frontend with API URL

Open `frontend/index.html` and update:

```js
// Before:
const API_URL = "YOUR_API_GATEWAY_URL/predict";

// After:
const API_URL =
  "https://3wnsbonc51.execute-api.ap-south-1.amazonaws.com/prod/predict";
```

### Step 4 — Host Frontend on S3

```bash
# Create S3 bucket
aws s3 mb s3://diabetesai-frontend --region ap-south-1

# Disable Block Public Access
aws s3api put-public-access-block \
  --bucket diabetesai-frontend \
  --public-access-block-configuration \
  "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"

# Apply public read policy
aws s3api put-bucket-policy --bucket diabetesai-frontend --policy '{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "PublicReadGetObject",
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::diabetesai-frontend/*"
  }]
}'

# Upload frontend
aws s3 cp frontend/index.html s3://diabetesai-frontend/

# Enable static website hosting
aws s3 website s3://diabetesai-frontend --index-document index.html
```

### ✅ Deployment Complete

| Resource     | URL                                                                  |
| ------------ | -------------------------------------------------------------------- |
| **Live App** | http://diabetesai-frontend.s3-website.ap-south-1.amazonaws.com       |
| **API**      | https://3wnsbonc51.execute-api.ap-south-1.amazonaws.com/prod/predict |

### Update Endpoint (If Retrained)

No redeployment needed — just update the Lambda environment variable:

```
AWS Console → Lambda → diabetes-prediction-api
→ Configuration → Environment Variables → Edit
→ ENDPOINT_NAME = <your-new-endpoint-name>
→ Save
```

---

## 📡 API Reference

### `POST /predict`

**Endpoint:**

```
https://3wnsbonc51.execute-api.ap-south-1.amazonaws.com/prod/predict
```

**Request:**

```http
POST /predict HTTP/1.1
Content-Type: application/json
```

```json
{
  "pregnancies": 6,
  "glucose": 148,
  "bloodPressure": 72,
  "skinThickness": 35,
  "insulin": 125,
  "bmi": 33.6,
  "dpf": 0.627,
  "age": 50
}
```

**Response — Diabetic:**

```json
{
  "prediction": 1,
  "probability": 0.7823,
  "label": "Diabetic",
  "raw_input": {
    "pregnancies": 6,
    "glucose": 148,
    "bloodPressure": 72,
    "skinThickness": 35,
    "insulin": 125,
    "bmi": 33.6,
    "dpf": 0.627,
    "age": 50
  }
}
```

**Response — Non-Diabetic:**

```json
{
  "prediction": 0,
  "probability": 0.2341,
  "label": "Non-Diabetic",
  "raw_input": { ... }
}
```

**cURL Test:**

```bash
curl -X POST https://3wnsbonc51.execute-api.ap-south-1.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 6,
    "glucose": 148,
    "bloodPressure": 72,
    "skinThickness": 35,
    "insulin": 125,
    "bmi": 33.6,
    "dpf": 0.627,
    "age": 50
  }'
```

**Error Response:**

```json
{
  "error": "An error occurred (ModelError) when calling InvokeEndpoint..."
}
```

---

## 🎨 Frontend UI Features

- **Dark clinical dashboard** — professional dark theme for medical contexts
- **Sidebar navigation** — live model metrics (Accuracy, F1, AUC-ROC, Recall)
- **8 interactive sliders** — real clinical ranges, units, and hints per feature
- **Animated SVG donut gauge** — probability with smooth cubic-bezier animation
- **Color-coded result card** — red for high risk, green for low risk
- **3-metric panel** — Prediction label, Confidence %, Risk Level
- **Animated probability bar** — fills on result with easing animation
- **CSV payload preview** — exact CSV sent to SageMaker, one-click copy
- **Prediction history table** — timestamps, key values, result badge
- **Demo mode** — simulates results before API URL is configured
- **Fully responsive** — adapts layout for mobile viewports
- **Hosted on S3** — no server needed, globally accessible

---

## 🔒 Security Considerations

> ⚠️ The SageMaker endpoint URL is never exposed in the frontend. The Lambda + API Gateway pattern keeps all AWS credentials server-side.

| Area               | Implementation                                                              |
| ------------------ | --------------------------------------------------------------------------- |
| **SageMaker Auth** | IAM Role — only Lambda has `sagemaker:InvokeEndpoint` permission            |
| **API Security**   | API Gateway provides rate limiting and DDoS protection                      |
| **Credentials**    | No AWS keys stored in frontend code — ever                                  |
| **CORS**           | `Access-Control-Allow-Origin: *` (restrict to your domain in production)    |
| **Audit Trail**    | CloudWatch logs all Lambda invocations with timestamps                      |
| **S3 Dataset**     | Block Public Access enabled on data bucket (only frontend bucket is public) |

---

## 🔮 Future Enhancements

### MLOps

- [ ] **SageMaker Pipelines** — automated retraining when new data arrives in S3
- [ ] **SageMaker Model Monitor** — data drift and quality degradation alerts
- [ ] **SageMaker Model Registry** — model versioning and approval workflows
- [ ] **SMOTE oversampling** in preprocessing to handle class imbalance

### Application

- [ ] **Amazon Cognito** — user authentication and login
- [ ] **Amazon DynamoDB** — persistent prediction history storage
- [ ] **Batch prediction dashboard** — upload CSV, download bulk predictions
- [ ] **SHAP explainability** — feature importance per prediction
- [ ] **Lower classification threshold** (0.5 → 0.35) to further boost Recall

### Infrastructure

- [ ] **CloudFront CDN** — global low-latency delivery for the frontend
- [ ] **AWS WAF** — Web Application Firewall on API Gateway
- [ ] **CI/CD pipeline** — AWS CodePipeline for automatic Lambda redeployment
- [ ] **Custom domain** — Route 53 + ACM certificate for HTTPS frontend

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only.**

It is **not** a substitute for professional medical diagnosis, advice, or treatment. The predictions made by this model should never be used as the sole basis for any clinical decision. Always consult a qualified healthcare professional for medical evaluation and diagnosis.

---

## 👤 Project Info

**Built with:** Amazon SageMaker Canvas · Data Wrangler · S3 · Lambda · API Gateway · SAM · CloudWatch · IAM

**Region:** ap-south-1 (Mumbai, India)

**Dataset:** Pima Indians Diabetes Dataset — 768 records · 8 features · Binary classification

---

_© 2026 DiabetesAI — End-to-End AWS ML Pipeline Project_
