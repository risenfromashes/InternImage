#!/bin/bash

# --- Configuration ---
FUNCTION_NAME="InferenceAPI"
ZIP_FILE="function.zip"
SOURCE_FILE="lambda_function.py"

# --- Step 1: Verification ---
# Check if the python file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: $SOURCE_FILE not found in the current directory."
    exit 1
fi

echo "Deploying $FUNCTION_NAME..."

# --- Step 2: Clean up old zip files ---
if [ -f "$ZIP_FILE" ]; then
    echo "Removing old zip file..."
    rm "$ZIP_FILE"
fi

# --- Step 3: Zip the Python file ---
echo "Zipping $SOURCE_FILE..."
# -r recurses (useful if you add folders later), -j junks paths (keeps zip flat)
zip -r "$ZIP_FILE" "$SOURCE_FILE"

# --- Step 4: Upload to AWS Lambda ---
echo "Uploading to AWS..."
aws lambda update-function-code \
    --function-name "$FUNCTION_NAME" \
    --zip-file "fileb://$ZIP_FILE" \
    --no-cli-pager

# --- Step 5: Check result ---
if [ $? -eq 0 ]; then
    echo "---------------------------------"
    echo "✅ Success! $FUNCTION_NAME has been updated."
    echo "---------------------------------"
else
    echo "---------------------------------"
    echo "❌ Upload failed. Please check your AWS credentials and function name."
    echo "---------------------------------"
fi