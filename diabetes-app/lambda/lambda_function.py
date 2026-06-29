import json
import boto3
import os

# SageMaker Runtime client
sagemaker_runtime = boto3.client(
    "sagemaker-runtime",
    region_name=os.environ.get("AWS_REGION", "ap-south-1")
)

ENDPOINT_NAME = os.environ.get(
    "ENDPOINT_NAME",
    "canvas-new-deployment-06-29-2026-12-09-AM/invocations"
)

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
    "Content-Type": "application/json"
}


def lambda_handler(event, context):
    # Handle CORS preflight
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

    try:
        body = json.loads(event.get("body", "{}"))

        # Extract features in the correct order:
        # Pregnancies, Glucose, BloodPressure, SkinThickness,
        # Insulin, BMI, DiabetesPedigreeFunction, Age
        features = [
            str(body.get("pregnancies", 0)),
            str(body.get("glucose", 120)),
            str(body.get("bloodPressure", 72)),
            str(body.get("skinThickness", 29)),
            str(body.get("insulin", 125)),
            str(round(float(body.get("bmi", 32.0)), 1)),
            str(round(float(body.get("dpf", 0.47)), 3)),
            str(body.get("age", 33)),
        ]

        payload = ",".join(features)

        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=payload,
            Accept="application/json"
        )

        result = json.loads(response["Body"].read().decode("utf-8"))

        # Parse SageMaker Canvas response format
        prediction = None
        probability = None

        if isinstance(result, dict):
            if "predictions" in result:
                pred = result["predictions"][0]
                prediction = pred.get("predicted_label", pred)
                probability = pred.get("probability", None)
            elif "predicted_label" in result:
                prediction = result["predicted_label"]
                probability = result.get("probability", None)
        else:
            prediction = result

        is_diabetic = int(float(str(prediction))) == 1
        prob = float(probability) if probability is not None else (0.78 if is_diabetic else 0.22)

        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps({
                "prediction": int(is_diabetic),
                "probability": round(prob, 4),
                "label": "Diabetic" if is_diabetic else "Non-Diabetic",
                "input": {k: body.get(k) for k in [
                    "pregnancies","glucose","bloodPressure",
                    "skinThickness","insulin","bmi","dpf","age"
                ]}
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": str(e)})
        }
