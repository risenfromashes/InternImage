import os
import time
import sys
import argparse
import requests
import boto3
import cv2
import numpy as np
import torch
import mmcv
import warnings
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

# --- SILENCE & SETUP ---
warnings.filterwarnings("ignore")
os.environ["TORCH_LOGS"] = "-all"

# Import Custom Ops if present (InternImage/MMSeg specific)
try:
    import mmcv_custom
    import mmseg_custom
except ImportError:
    pass

from mmseg.apis import init_segmentor, inference_segmentor

# --- CONFIGURATION ---
TABLE_NAME = os.environ.get("TABLE_NAME", "InferenceJobs")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
BUCKET = os.environ.get("INPUT_BUCKET")
WEBHOOK_API_KEY = os.environ.get("LAMBDA_API_KEY")

DEFAULT_CONFIG = "model/config.py"
DEFAULT_CHECKPOINT = "model/model.pth"

# AWS Clients
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(TABLE_NAME)
s3 = boto3.client("s3")

WEBHOOK_HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": WEBHOOK_API_KEY,
}


class Segmentor:
    """Wrapper for MMSegmentation Inference with Custom Pipeline."""

    def __init__(self, config_path, checkpoint_path, device="cuda:0"):
        print(f"--- Loading Model: {config_path} ---")

        self.cfg = mmcv.Config.fromfile(config_path)

        # Inject Custom Pipeline
        img_norm_cfg = self.cfg.img_norm_cfg
        self.cfg.data.test.pipeline = [
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=None,
                img_ratios=1.0,
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="ResizeToMultiple", size_divisor=32),
                    dict(type="RandomFlip", prob=0.0),
                    dict(type="Normalize", **img_norm_cfg),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ]

        self.cfg.model.test_cfg.mode = "whole"  # Force whole image inference

        self.device = device
        self.model = init_segmentor(self.cfg, checkpoint=None, device=self.device)

        # Load weights safely
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state, strict=False)
        self.model.cuda()
        self.model.eval()
        self.amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        print("Model Loaded & Pipeline Configured.")

    def predict(self, img_path):
        """Returns binary mask (numpy array)."""
        with torch.no_grad(), self.amp_ctx:
            result = inference_segmentor(self.model, img_path)
        return result[0]


class JobManager:
    """Handles AWS Interactions: S3, DynamoDB, and Webhooks."""

    @staticmethod
    def get_pending_jobs(limit=1000):
        try:
            resp = table.query(
                IndexName="StatusIndex",
                KeyConditionExpression=Key("status").eq("PENDING"),
                Limit=limit,
            )
            return resp.get("Items", [])
        except Exception as e:
            print(f"DB Query Error: {e}")
            return []

    @staticmethod
    def lock_job(job_id):
        try:
            table.update_item(
                Key={"job_id": job_id},
                UpdateExpression="set #s = :s",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={":s": "RUNNING"},
                ConditionExpression="attribute_exists(job_id)",
            )
            return True
        except Exception:
            return False

    @staticmethod
    def download_input(bucket, s3_key, local_path):
        """Downloads image and returns metadata (if any)."""
        obj = s3.get_object(Bucket=bucket, Key=s3_key)

        with open(local_path, "wb") as f:
            f.write(obj["Body"].read())

        # Extract Webhook URL from Metadata (Case-insensitive check)
        meta = obj.get("Metadata", {})
        webhook = meta.get("webhook_url") or meta.get("callback_url")

        return webhook

    @staticmethod
    def upload_result(bucket, job_id, mask_array):
        """Encodes mask to PNG and uploads to S3."""
        res_key = f"results/{job_id}.png"
        mask_uint8 = (mask_array.astype(np.uint8)) * 255
        _, buf = cv2.imencode(".png", mask_uint8)

        s3.put_object(
            Bucket=bucket, Key=res_key, Body=buf.tobytes(), ContentType="image/png"
        )
        return res_key

    @staticmethod
    def mark_complete(job_id, result_key, bucket, request_ts, webhook_url=None):
        """Updates DynamoDB and fires Webhook with full payload."""
        completion_ts = int(time.time())

        # 1. Update DB
        table.update_item(
            Key={"job_id": job_id},
            UpdateExpression="set #s = :s, result_key = :r, completed_at = :c",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":s": "SUCCESS",
                ":r": result_key,
                ":c": completion_ts,
            },
        )

        # 2. Fire Webhook
        if webhook_url:
            try:
                # Generate Presigned URL for the payload
                mask_url = s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket, "Key": result_key},
                    ExpiresIn=3600,  # 1 Hour
                )

                payload = {
                    "job_id": job_id,
                    "status": "SUCCESS",
                    "mask_url": mask_url,
                    "s3_path": result_key,
                    "request_timestamp": request_ts,
                    "job_completion_timestamp": completion_ts,
                }

                requests.post(
                    webhook_url, json=payload, timeout=5, headers=WEBHOOK_HEADERS
                )
                print(f"Webhook sent to {webhook_url}")
            except Exception as e:
                print(f"Webhook failed: {e}")

    @staticmethod
    def mark_failed(job_id, error_msg, webhook_url=None):
        """Updates DynamoDB and fires Webhook on error."""
        table.update_item(
            Key={"job_id": job_id},
            UpdateExpression="set #s = :s, error_msg = :e",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":s": "FAILED", ":e": str(error_msg)},
        )

        if webhook_url:
            try:
                requests.post(
                    webhook_url,
                    json={
                        "job_id": job_id,
                        "status": "FAILED",
                        "error": str(error_msg),
                        "job_completion_timestamp": int(time.time()),
                    },
                    timeout=5,
                    headers=WEBHOOK_HEADERS,
                )
            except:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    if not BUCKET:
        print("Error: INPUT_BUCKET env var missing.")
        return

    # 1. Load Model
    try:
        model = Segmentor(args.config, args.checkpoint)
    except Exception as e:
        print(f"CRITICAL: Model load failed. {e}")
        return

    print("Worker Ready. Polling Queue...")

    # 2. Processing Loop
    idle_strikes = 0

    while True:
        jobs = JobManager.get_pending_jobs()

        if not jobs:
            print(f"Queue Empty. Strike {idle_strikes+1}/3")
            idle_strikes += 1
            if idle_strikes >= 3:
                print("Shutting down worker.")
                break
            time.sleep(30)
            continue

        idle_strikes = 0
        print(f"Processing Batch: {len(jobs)} jobs found.")

        for job in jobs:
            job_id = job["job_id"]
            s3_key = job["s3_input"]
            request_ts = job.get("created_at")  # Fetch timestamp from DB item
            request_ts = int(request_ts) if request_ts else None

            local_path = f"/tmp/{job_id}.jpg"
            webhook_url = None

            print(f"--> Job: {job_id}")

            # Try to acquire lock
            if not JobManager.lock_job(job_id):
                print(f"Skipping {job_id} (Locked/Deleted)")
                continue

            try:
                # A. Download
                webhook_url = JobManager.download_input(BUCKET, s3_key, local_path)

                # B. Infer (Pass path directly for LoadImageFromFile)
                mask = model.predict(local_path)

                # C. Upload
                result_key = JobManager.upload_result(BUCKET, job_id, mask)

                # D. Complete
                JobManager.mark_complete(
                    job_id=job_id,
                    result_key=result_key,
                    bucket=BUCKET,
                    request_ts=request_ts,
                    webhook_url=webhook_url,
                )

            except Exception as e:
                print(f"Failed {job_id}: {e}")
                JobManager.mark_failed(job_id, e, webhook_url)

            finally:
                if os.path.exists(local_path):
                    os.remove(local_path)


if __name__ == "__main__":
    main()
