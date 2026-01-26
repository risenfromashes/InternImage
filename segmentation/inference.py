import warnings
import os
# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TORCH_LOGS"] = "-all"


import time
import torch
import mmcv
import argparse
import cv2
import numpy as np


try:
    import mmcv_custom
    import mmseg_custom
except ImportError:
    pass

from mmseg.apis import init_segmentor, inference_segmentor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--image', default='deploy/demo.png')
    args = parser.parse_args()

    # 1. Load Config & Force Whole Mode
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.test_cfg.mode = 'whole' 
    # Ensure internal resize is compatible with backbone (align to 32)

    # 2. Init Model
    print("--- Loading Model ---")
    model = init_segmentor(cfg, checkpoint=None, device='cuda:0')
    
    # 3. Load Weights (Safe)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state, strict=False)
    
    model.cuda()
    model.eval()
    
    # 4. Prepare Image
    img = mmcv.imread(args.image)
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16)

    # 5. Warmup (TRIGGERS COMPILATION)
    print("--- Warmup & Compilation (This may take 1 minute)... ---")
    start_warmup = time.time()
    with torch.no_grad(), amp_ctx:
        _ = inference_segmentor(model, img)
    torch.cuda.synchronize()
    print(f"Warmup finished in {time.time() - start_warmup:.2f}s")

    # 6. Benchmark
    print("--- Running Timed Inference ---")
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad(), amp_ctx:
        result = inference_segmentor(model, img)
    
    torch.cuda.synchronize()
    end = time.time()
    duration = end - start
    
    print(f"\nSUCCESS!")
    print(f"Mode:          Whole Inference + torch.compile")
    print(f"Inference Time: {duration:.4f}s")
    print(f"FPS:           {1/duration:.2f}")

    # 7. Save
    mask = result[0]
    overlay = np.zeros_like(img)
    overlay[mask == 1] = [0, 255, 0] 
    final = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    cv2.imwrite('result_compiled.jpg', final)
    print("Saved to result_compiled.jpg")

if __name__ == '__main__':
    main()

