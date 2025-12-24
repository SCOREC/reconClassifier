import time
import torch
import numpy as np
import argparse
import sys
import os
from pathlib import Path

# Ensure we can import from the current directory and pgkylFrontEnd (via PYTHONPATH)
sys.path.append(str(Path(__file__).parent))

try:
    from XPointMLTest import UNet, loadPgkylDataFromCache, cachedPgkylDataExists
    from utils import auxFuncs, gkData
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you have sourced envPyTorch.sh and are running with correct PYTHONPATH.")
    sys.exit(1)

def compare_hessian_vs_ml(param_file, cache_dir, model_path, frame_list, device='cuda'):
    print(f"Comparing Hessian vs ML on frames {frame_list}")
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Param File: {param_file}")
    print(f"Cache Dir: {cache_dir}")

    # Load Model
    # UNet signature: def __init__(self, input_channels=4, base_channels=32, *, dropout_rate):
    model = UNet(input_channels=4, base_channels=32, dropout_rate=0.15).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
        
    model.eval()
    
    hessian_times = []
    ml_times = []
    
    # Warmup
    dummy_input = torch.randn(1, 4, 1024, 1024).to(device)
    with torch.no_grad():
        _ = model(dummy_input)

    print(f"{'Frame':<10} | {'Hessian (s)':<15} | {'ML (s)':<15} | {'Speedup':<10}")
    print("-" * 60)

    cache_path = Path(cache_dir) if cache_dir else None

    for fnum in frame_list:
        try:
            psi = None
            dx = None
            
            # Try loading from cache first
            if cache_path and cachedPgkylDataExists(cache_path, fnum, "psi"):
                fields_to_load = {"psi": None, "coords": None}
                loaded = loadPgkylDataFromCache(cache_path, fnum, fields_to_load)
                psi = loaded["psi"]
                coords = loaded["coords"]
                # Calculate dx from coords
                dx = [c[1] - c[0] for c in coords]
            else:
                # Fallback to gkData (might fail if getData.py is buggy)
                params = {}
                params["polyOrderOverride"] = 0
                var = gkData.gkData(str(param_file), fnum, 'psi', params).compactRead()
                psi = var.data
                dx = var.dx
            
            if psi is None:
                print(f"Could not load data for frame {fnum}")
                continue

            # --- Measure Hessian Time ---
            t0 = time.time()
            
            # Replicating Hessian logic
            critPoints = auxFuncs.getCritPoints(psi)
            [xpts, optsMax, optsMin] = auxFuncs.getXOPoints(psi, critPoints)
            
            t1 = time.time()
            hessian_time = t1 - t0
            hessian_times.append(hessian_time)
            
            # --- Measure ML Time ---
            # Preprocess - Calculate derived fields
            [df_dx,df_dy,df_dz] = auxFuncs.genGradient(psi,dx)
            [d2f_dxdx,d2f_dxdy,d2f_dxdz] = auxFuncs.genGradient(df_dx,dx)
            [d2f_dydx,d2f_dydy,d2f_dydz] = auxFuncs.genGradient(df_dy,dx)
            bx = df_dy
            by = -df_dx
            # mu0 is usually 1.0 in normalized units or available in var.mu0
            # If we loaded from cache, we don't have var.mu0. 
            # Assuming mu0=1.0 for now as it's common in normalized simulations, 
            # or we could read it from param file, but let's stick to 1.0 or check if we can get it.
            # In XPointMLTest.py: jz = -(d2f_dxdx + d2f_dydy) / var.mu0
            # In getConst.py: self.mu0 = mu0 (from param file).
            # Let's assume mu0=1.0 to avoid reading param file again, or just use 1.0.
            mu0 = 1.0 
            jz = -(d2f_dxdx + d2f_dydy) / mu0
            
            # Normalize (using same logic as XPointMLTest.py)
            psi_norm = (psi - psi.mean()) / (psi.std() + 1e-8)
            bx_norm = (bx - bx.mean()) / (bx.std() + 1e-8)
            by_norm = (by - by.mean()) / (by.std() + 1e-8)
            jz_norm = (jz - jz.mean()) / (jz.std() + 1e-8)
            
            # Stack
            psi_torch = torch.from_numpy(psi_norm).float().unsqueeze(0)
            bx_torch = torch.from_numpy(bx_norm).float().unsqueeze(0)
            by_torch = torch.from_numpy(by_norm).float().unsqueeze(0)
            jz_torch = torch.from_numpy(jz_norm).float().unsqueeze(0)
            
            input_tensor = torch.cat((psi_torch, bx_torch, by_torch, jz_torch)).unsqueeze(0).to(device)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            t2 = time.time()
            
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output)
                mask = (prob > 0.5).float()
                
            if device == 'cuda':
                torch.cuda.synchronize()
            t3 = time.time()
            
            ml_time = t3 - t2
            ml_times.append(ml_time)
            
            print(f"{fnum:<10} | {hessian_time:<15.4f} | {ml_time:<15.4f} | {hessian_time/ml_time:<10.2f}")
            
        except Exception as e:
            print(f"Error processing frame {fnum}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if hessian_times and ml_times:
        avg_hessian = np.mean(hessian_times)
        avg_ml = np.mean(ml_times)
        
        print("\n" + "="*60)
        print(f"Average Hessian Time: {avg_hessian:.4f}s")
        print(f"Average ML Time:      {avg_ml:.4f}s")
        print(f"Average Speedup:      {avg_hessian/avg_ml:.2f}x")
        print("="*60)
    else:
        print("No frames processed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Hessian-based vs ML-based X-point detection performance.")
    parser.add_argument('--paramFile', type=str, required=True, help="Path to the parameter file")
    parser.add_argument('--xptCacheDir', type=str, default=None, help="Path to cache directory (optional)")
    parser.add_argument('--modelPath', type=str, required=True, help="Path to the trained model checkpoint (.pt)")
    parser.add_argument('--frames', type=str, default="141-150", help="Range of frames (e.g., '141-150' or '141,142,143')")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run ML model on")
    
    args = parser.parse_args()
    
    # Parse frames
    if '-' in args.frames:
        start, end = map(int, args.frames.split('-'))
        frames = range(start, end + 1)
    else:
        frames = [int(x) for x in args.frames.split(',')]
        
    compare_hessian_vs_ml(args.paramFile, args.xptCacheDir, args.modelPath, frames, args.device)
