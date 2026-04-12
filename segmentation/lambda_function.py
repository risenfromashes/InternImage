import json
import boto3
import uuid
import base64
import os
import time
from botocore.exceptions import ClientError

# --- INFRASTRUCTURE CONFIG ---
s3 = boto3.client('s3')
batch = boto3.client('batch')
dynamodb = boto3.resource('dynamodb')

# Configuration Constants
BUCKET_NAME = os.environ.get('BUCKET_NAME', "pavement-data")
MODEL_URI = os.environ.get('MODEL_URI', 's3://limbics-ml-models/internimage/')
TABLE_NAME = os.environ.get('TABLE_NAME', 'InferenceJobs')
JOB_QUEUE = os.environ.get('JOB_QUEUE', "GPU-Queue")
JOB_DEF = os.environ.get('JOB_DEF', "GPU-Job")
API_KEY_SECRET = os.environ.get('API_KEY')
WEBHOOK_URL = os.environ.get('WEBHOOK_URL')
SELF_URL = os.environ.get('SELF_URL')

# Initialize Table Resource
table = dynamodb.Table(TABLE_NAME)

# --- HELPER FUNCTIONS ---

def get_response(status_code, body):
    """Standardized API Gateway response format."""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(body) if isinstance(body, (dict, list)) else body
    }

def check_auth(event):
    """Validates the API Key from headers (Case Insensitive)."""
    if not API_KEY_SECRET:
        return False
    
    headers = event.get('headers', {})
    headers_lower = {k.lower(): v for k, v in headers.items()}
    client_key = headers_lower.get('x-api-key')
    
    return client_key == API_KEY_SECRET

def ensure_worker_active(force=False):
    """Checks if GPU worker is running. If cold or stale (>15 mins) or forced, triggers it."""
    try:
        current_time = int(time.time())
        lock_item = table.get_item(Key={'job_id': 'SYSTEM_LOCK'}).get('Item', {})
        
        status = lock_item.get('status')
        last_active = int(lock_item.get('timestamp', 0))
        is_stale = (current_time - last_active) > 1000
        
        if force or status != 'RUNNING' or is_stale:
            reason = "Force" if force else f"Status {status}, Stale={is_stale}"
            print(f"System Check: {reason}. Starting Worker...")
            
            table.put_item(Item={
                'job_id': 'SYSTEM_LOCK',
                'status': 'RUNNING',
                'timestamp': current_time
            })
            
            batch.submit_job(
                jobName=f"drainer-{current_time}",
                jobQueue=JOB_QUEUE,
                jobDefinition=JOB_DEF,
                containerOverrides={
                    'environment': [
                        {'name': 'INPUT_BUCKET', 'value': BUCKET_NAME},
                        {'name': 'MODEL_URI', 'value': MODEL_URI}
                    ]
                }
            )
            return True
        return False
    except Exception as e:
        print(f"Error ensuring worker active: {e}")
        return False

# --- HANDLERS ---

def handle_inference_request(body):
    """(POST /) Handles Image Submission for Inference."""
    image_b64 = body.get('image_base64')
    if not image_b64:
        return get_response(400, {'error': 'Missing image_base64'})

    job_id = str(uuid.uuid4())
    s3_key = f"inputs/{job_id}.jpg"

    retry_url = f"{SELF_URL}/?action=ensure_worker&force=true"

    try:
        s3.put_object(
            Bucket=BUCKET_NAME, 
            Key=s3_key, 
            Body=base64.b64decode(image_b64),
            Metadata={
                'webhook_url': str(WEBHOOK_URL),
                'retry_url': retry_url
            }
        )

        table.put_item(Item={
            'job_id': job_id,
            'status': 'PENDING',
            's3_input': s3_key,
            'created_at': int(time.time())
        })

        ensure_worker_active()

        return get_response(200, {'job_id': job_id, 'status': 'QUEUED'})

    except Exception as e:
        print(f"Inference POST Error: {e}")
        return get_response(500, {'error': str(e)})

def handle_completion_webhook(body):
    """(POST /frame-segmentation) Logs completion details."""
    try:
        # Log the entire body for debugging/audit
        print("Received Completion Webhook:", json.dumps(body))

        return get_response(200, {'message': 'Completion received and logged'})
    except Exception as e:
        print(f"Webhook Error: {e}")
        return get_response(500, {'error': 'Failed to process webhook'})

def handle_get_status(query_params):
    """(GET /) Handles Status Check."""
    job_id = query_params.get('job_id')
    
    # Check for Action Flag (Worker Handover)
    action = query_params.get('action')
    if action == 'ensure_worker':
        force_start = query_params.get('force', 'false').lower() == 'true'
        ensure_worker_active(force=force_start)
        return get_response(200, {'message': 'Worker check triggered', 'forced': force_start})

    if not job_id:
        return get_response(400, {'error': 'Missing job_id'})

    item = table.get_item(Key={'job_id': job_id}).get('Item')
    if not item:
        return get_response(404, {'error': 'Job not found'})

    status = item['status']
    
    # Fail-safe: If user checks a PENDING job, ensure worker is alive
    if status == 'PENDING':
        ensure_worker_active()

    response_data = {'job_id': job_id, 'status': status}

    if status == 'SUCCESS' and 'result_key' in item:
        try:
            url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': BUCKET_NAME, 'Key': item['result_key']},
                ExpiresIn=36000
            )
            response_data['result_url'] = url
        except ClientError:
            pass 
    
    return get_response(200, response_data)

# --- MAIN ENTRY POINT ---

def lambda_handler(event, context):
    # 1. Auth Check
    if not check_auth(event):
        return get_response(403, {'error': 'Forbidden: Invalid API Key'})

    # 2. Extract Request Details
    http_method = event.get('requestContext', {}).get('http', {}).get('method')
    # Support both HTTP API v2 (rawPath) and REST API v1 (path)
    path = event.get('rawPath') or event.get('path') or '/'
    
    # 3. Router
    if http_method == 'POST':
        try:
            body = json.loads(event.get('body', '{}'))
            
            # ROUTE: /frame-segmentation
            if path == '/frame-segmentation':
                return handle_completion_webhook(body)
            
            # ROUTE: Default (root) or explicit /inference
            else:
                return handle_inference_request(body)
                
        except json.JSONDecodeError:
            return get_response(400, {'error': 'Invalid JSON body'})

    elif http_method == 'GET':
        query_params = event.get('queryStringParameters', {}) or {}
        return handle_get_status(query_params)
    
    return get_response(405, {'error': 'Method Not Allowed'})