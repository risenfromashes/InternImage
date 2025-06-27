# video_demo.py
import torch
import numpy as np
import cv2
import mmcv
import mmcv_custom  # noqa: F401,F403
import mmseg_custom  # noqa: F401,F403
import os
from argparse import ArgumentParser
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor, inference_segmentor
from tqdm import tqdm

def parse_args():
    parser = ArgumentParser(description="Run crack-segmentation on a video")
    parser.add_argument('--input',     required=True, help='Input video file')
    parser.add_argument('--config',    required=True, help='Mask2Former config .py')
    parser.add_argument('--checkpoint',required=True, help='Trained model .pth')
    parser.add_argument('--output-dir', required=True, help='Output directory for video and frames')
    parser.add_argument('--device',    default='cuda:0', help='Inference device')
    parser.add_argument(
        '--opacity',
        type=float, default=0.5,
        help='Overlay opacity (0.0 – 1.0)')
    return parser.parse_args()

def setup_model(config_path, checkpoint_path, device):
    """Initialize and setup the segmentation model."""
    # 1) init model (loads palette & classes from config/checkpoint meta)
    model = init_segmentor(config_path, checkpoint=None, device=device)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 2) extract state_dict and meta
    state_dict = ckpt.get('state_dict', ckpt)
    meta = ckpt.get('meta', {})

    # 3) load weights & attach CLASSES/PALETTE
    model.load_state_dict(state_dict, strict=False)
    if 'CLASSES' in meta:
        model.CLASSES = meta['CLASSES']
    if 'PALETTE' in meta:
        model.PALETTE = meta['PALETTE']

    # 4) optimize model for inference
    model = model.half()
    model.eval()
    
    return model

def setup_video_io(input_path, output_path):
    """Setup video input and output streams."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    return cap, writer, fps, total_frames

def process_frame_batch(frame_batch, model, args, writer, frames_dir, start_frame_idx, pbar):
    """Process a batch of frames and save results."""
    with torch.no_grad(), torch.cuda.amp.autocast():
        results = inference_segmentor(model, frame_batch)
    
    for i, result in enumerate(results):
        frame_idx = start_frame_idx + i
        frame = frame_batch[i]
        
        # Create mask
        mask = np.zeros_like(frame, dtype=np.uint8)
        mask[result == 1] = (0, 255, 0)
        
        # Create overlay
        overlay = cv2.addWeighted(frame, 1 - args.opacity, mask, args.opacity, 0)
        
        # Save video frame
        writer.write(overlay)
        
        # Save image frame
        frame_name = f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(os.path.join(frames_dir, frame_name), overlay)
        
        # Update progress
        pbar.update(1)

def main():
    args = parse_args()

    # Create output directory structure
    output_dir = args.output_dir
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    
    # Set output video path
    video_output_path = os.path.join(output_dir, 'segmented_video.mp4')
    print(f"Output directory: {output_dir}")
    print(f"Overlay frames will be saved to: {frames_dir}")
    print(f"Video will be saved to: {video_output_path}")

    # Setup model
    model = setup_model(args.config, args.checkpoint, args.device)

    # Setup video I/O
    cap, writer, fps, total_frames = setup_video_io(args.input, video_output_path)
    print(f"Processing {total_frames} frames at {fps:.2f} FPS...")

    # 3) process frames
    frame_idx = 0
    frame_batch = []
    batch_size = 4
    
    with tqdm(total=total_frames, desc="Processing video", unit="frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Process remaining frames in batch if any
                if frame_batch:
                    start_idx = frame_idx - len(frame_batch)
                    process_frame_batch(frame_batch, model, args, writer, frames_dir, start_idx, pbar)
                break
            
            frame_batch.append(frame)
            frame_idx += 1
            
            # Process batch when it reaches batch_size
            if len(frame_batch) == batch_size:
                start_idx = frame_idx - batch_size
                process_frame_batch(frame_batch, model, args, writer, frames_dir, start_idx, pbar)
                frame_batch = []  # Clear batch

    cap.release()
    writer.release()
    print(f'Done! Saved overlay video to: {video_output_path}')
    print(f'Saved {frame_idx} overlay frames to: {frames_dir}')

if __name__ == '__main__':
    main()
