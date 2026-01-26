import requests
import base64
import time
import os
import json
from dotenv import load_dotenv

# 1. Load Config
load_dotenv()
API_KEY = os.environ.get("LAMBDA_API_KEY")
LAMBDA_ENDPOINT = os.environ.get("LAMBDA_ENDPOINT")
IMG_PATH = "deploy/demo.jpg"

# Validation
if not API_KEY:
    print("❌ Error: LAMBDA_API_KEY not found in .env")
    exit(1)
if not LAMBDA_ENDPOINT:
    print("❌ Error: LAMBDA_ENDPOINT not found in .env")
    exit(1)

# Clean URL (remove trailing slash if present to avoid double // issues)
BASE_URL = LAMBDA_ENDPOINT.rstrip("/")

# Global Headers
HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY,
}


def submit_inference_job(image_path):
    """Submits an image to the root endpoint '/'"""
    print(f"\n--- 🚀 Submitting Inference Job ---")

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    # Target: Root URL
    url = f"{BASE_URL}/"

    try:
        res = requests.post(url, json={"image_base64": b64}, headers=HEADERS)

        if res.status_code == 200:
            job_id = res.json().get("job_id")
            print(f"✅ Job Submitted. ID: {job_id}")
            return job_id
        else:
            print(f"❌ Submission Failed [{res.status_code}]: {res.text}")
            return None
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return None


def poll_job_status(job_id):
    """Polls the root endpoint for status"""
    print(f"\n--- 🕵️ Polling Status for {job_id} ---")
    url = f"{BASE_URL}/"

    while True:
        try:
            res = requests.get(url, params={"job_id": job_id}, headers=HEADERS)

            if res.status_code != 200:
                print(f"❌ Polling Error: {res.text}")
                break

            data = res.json()
            status = data.get("status")
            print(f"Status: {status}")

            if status == "SUCCEEDED":
                print(f"🎉 Result: {data.get('result_url')}")
                return
            elif status == "FAILED":
                print("💀 Job Failed!")
                return

        except Exception as e:
            print(f"❌ Polling Exception: {e}")
            break

        time.sleep(5)


def test_webhook_endpoint(job_id):
    """Simulates a worker sending a completion status to '/frame-segmentation'"""
    print(f"\n--- 🤖 Testing Webhook (Worker Simulation) ---")

    # Target: /frame-segmentation
    url = f"{BASE_URL}/frame-segmentation"

    payload = {
        "job_id": job_id,
        "status": "SUCCEEDED",
        "mask_url": "s3://bucket/fake_result.png",
        "s3_path": "inputs/fake_input.jpg",
        "request_timestamp": int(time.time()) - 60,
        "job_completion_timestamp": int(time.time()),
    }

    try:
        res = requests.post(url, json=payload, headers=HEADERS)
        print(f"Webhook Response [{res.status_code}]: {res.text}")
    except Exception as e:
        print(f"❌ Webhook Error: {e}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    job_id = submit_inference_job(IMG_PATH)

    test_webhook_endpoint(job_id)
