#!/bin/bash
set -e

# --- CONFIGURATION ---
# MODEL_URI should be a directory, e.g., s3://my-bucket/weights/internimage-v1/
S3_URI=${MODEL_URI} 

# Local directory where we will dump the S3 folder contents
LOCAL_MODEL_DIR="/data/models"

# Assumed filenames inside that directory
LOCAL_CONFIG="$LOCAL_MODEL_DIR/config.py"
LOCAL_CHECKPOINT="$LOCAL_MODEL_DIR/model.pth"

# 1. Ensure Local Directory Exists
mkdir -p "$LOCAL_MODEL_DIR"

echo "--- 🚀 Worker Initialization ---"

# 2. Check if model files exist (Simple check: if model.pth is missing, we download)
if [ -f "$LOCAL_CHECKPOINT" ]; then
    echo "✅ Model found locally at $LOCAL_CHECKPOINT. Skipping download."
else
    if [ -z "$S3_URI" ]; then
        echo "❌ Error: MODEL_URI environment variable is not set."
        exit 1
    fi

    echo "⬇️ Downloading recursive content from $S3_URI..."
    
    # --recursive copies the folder contents
    aws s3 cp "$S3_URI" "$LOCAL_MODEL_DIR" --recursive
    
    if [ $? -ne 0 ]; then
        echo "❌ Error downloading model directory."
        exit 1
    fi
    
    # Verify the specific files we need are actually there
    if [[ ! -f "$LOCAL_CONFIG" || ! -f "$LOCAL_CHECKPOINT" ]]; then
        echo "❌ Error: Download finished, but expected files (config.py, model.pth) are missing in $LOCAL_MODEL_DIR."
        ls -l "$LOCAL_MODEL_DIR"
        exit 1
    fi
fi

# 3. Activate Conda Environment
# We source conda.sh to ensure 'conda activate' works in the script
source /opt/conda/etc/profile.d/conda.sh
conda activate internimage

# 4. Run the Worker
echo "🔥 Starting Inference Worker..."
exec python inference_aws.py \
    --config "$LOCAL_CONFIG" \
    --checkpoint "$LOCAL_CHECKPOINT" \
    "$@"
