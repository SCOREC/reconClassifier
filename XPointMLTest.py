import numpy as np
import matplotlib.pyplot as plt
import os, errno
from pathlib import Path
import sys
import argparse

from utils import gkData
from utils import auxFuncs
from utils import plotParams

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision.transforms import v2 # rotate tensor

from torch.utils.data import DataLoader, Dataset

from timeit import default_timer as timer

# Import mixed precision training components
from torch.amp import autocast, GradScaler

from ci_tests import SyntheticXPointDataset, test_checkpoint_functionality

# Import benchmark module
from benchmark import TrainingBenchmark

# Import evaluation metrics module
from eval_metrics import ModelEvaluator, evaluate_model_on_dataset

def set_seed(seed):
    """
    Set random seed for reproducibility across all libraries
    
    Parameters:
    seed (int): Random seed value
    """
    if seed is None:
        return
    
    print(f"Setting random seed to {seed} for reproducibility")
    
    # Python random
    import random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # Make PyTorch deterministic (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def expand_xpoints_mask(binary_mask, kernel_size=9):
    """
    Expands each X-point in a binary mask to include surrounding cells
    in a square grid of size kernel_size x kernel_size.
    
    Parameters:
    binary_mask : numpy.ndarray
        2D binary mask with 1s at X-point locations
    kernel_size : int
        Size of the square grid (must be odd number)
        
    Returns:
    numpy.ndarray
        Expanded binary mask with 1s in kernel_size×kernel_size regions around X-points
    """
    
    # Get shape of the original mask
    h, w = binary_mask.shape
    
    # Create a copy to avoid modifying the original
    expanded_mask = np.zeros_like(binary_mask)
    
    # Find coordinates of all X-points
    x_points = np.argwhere(binary_mask > 0)
    
    # For each X-point, set a kernel_size×kernel_size area to 1
    half_size = kernel_size // 2
    for point in x_points:
        # Get the corner coordinates for the square centered at the X-point
        x_min = max(0, point[0] - half_size)
        x_max = min(h, point[0] + half_size + 1)
        y_min = max(0, point[1] - half_size)
        y_max = min(w, point[1] + half_size + 1)
        
        # Set the square area to 1
        expanded_mask[x_min:x_max, y_min:y_max] = 1
    
    return expanded_mask

def rotate(frameData,deg):
    if deg not in [90, 180, 270]:
        print(f"invalid rotation specified... exiting")
        sys.exit()
        
    psi = v2.functional.rotate(frameData["psi"], deg, v2.InterpolationMode.BILINEAR)
    all = v2.functional.rotate(frameData["all"], deg, v2.InterpolationMode.BILINEAR)
    # For mask, use nearest neighbor interpolation to preserve binary values
    mask = v2.functional.rotate(frameData["mask"], deg, v2.InterpolationMode.NEAREST)
    
    return {
        "fnum": frameData["fnum"],
        "rotation": deg,
        "reflectionAxis": -1, # no reflection
        "psi": psi,
        "all": all,
        "mask": mask,
        "x": frameData["x"],
        "y": frameData["y"],
        "filenameBase": frameData["filenameBase"],
        "params": frameData["params"]
    }

def reflect(frameData,axis):
    if axis not in [0,1]:
        print(f"invalid reflection axis specified... exiting")
        sys.exit()
            
    psi = torch.flip(frameData["psi"], dims=(axis+1,))
    all = torch.flip(frameData["all"], dims=(axis+1,))
    mask = torch.flip(frameData["mask"], dims=(axis+1,))
    
    return {
        "fnum": frameData["fnum"],
        "rotation": 0,
        "reflectionAxis": axis,
        "psi": psi,
        "all": all,
        "mask": mask,
        "x": frameData["x"],
        "y": frameData["y"],
        "filenameBase": frameData["filenameBase"],
        "params": frameData["params"]
    }

def getPgkylData(paramFile, frameNumber, verbosity):
  if verbosity > 0:
    print(f"=== frame {frameNumber} ===")
  params = {} #Initialize dictionary to store plotting and other parameters
  params["polyOrderOverride"] = 0 #Override default dg interpolation and interpolate to given number of points
  constrcutBandJ = 1
  #Read vector potential
  var = gkData.gkData(str(paramFile),frameNumber,'psi',params).compactRead()
  psi = var.data
  coords = var.coords
  axesNorm = var.d[ var.speciesFileIndex.index('ion') ]
  if verbosity > 0:
    print(f"psi shape: {psi.shape}, min={psi.min()}, max={psi.max()}")
  #Construct B and J (first and second derivatives)
  [df_dx,df_dy,df_dz] = auxFuncs.genGradient(psi,var.dx)
  [d2f_dxdx,d2f_dxdy,d2f_dxdz] = auxFuncs.genGradient(df_dx,var.dx)
  [d2f_dydx,d2f_dydy,d2f_dydz] = auxFuncs.genGradient(df_dy,var.dx)
  bx = df_dy
  by = -df_dx
  jz = -(d2f_dxdx + d2f_dydy) / var.mu0
  del df_dx,df_dy,df_dz,d2f_dxdx,d2f_dxdy,d2f_dxdz,d2f_dydx,d2f_dydy,d2f_dydz
  #Indicies of critical points, X points, and O points (max and min)
  critPoints = auxFuncs.getCritPoints(psi)
  [xpts, optsMax, optsMin] = auxFuncs.getXOPoints(psi, critPoints)
  return [var.filenameBase, axesNorm, critPoints, xpts, optsMax, optsMin, coords, psi, bx, by, jz]

def cachedPgkylDataExists(cacheDir, frameNumber, fieldName):
  if cacheDir == None:
      return False
  else:
      cachedFrame = cacheDir / f"{frameNumber}_{fieldName}.npy"
      return cachedFrame.exists();

def loadPgkylDataFromCache(cacheDir, frameNumber, fields):
  outFields = {}
  if cacheDir != None:
      for name in fields.keys():
        if name == "fileName":
            with open(cacheDir / f"{frameNumber}_{name}.txt", "r") as file:
                outFields[name] = file.read().rstrip()
        else:
            outFields[name] = np.load(cacheDir / f"{frameNumber}_{name}.npy")
      return outFields
  else:
      return None

def writePgkylDataToCache(cacheDir, frameNumber, fields):
  if cacheDir != None:
      for name, field in fields.items():
        if name == "fileName":
            with open(cacheDir / f"{frameNumber}_{name}.txt", "w") as text_file:
                text_file.write(f"{field}")
        else:
            np.save(cacheDir / f"{frameNumber}_{name}.npy",field)

# DATASET DEFINITION
class XPointDataset(Dataset):
    """
    Dataset that processes frames in [fnumList]. For each frame (fnum):
      - Sets up "params" according to your snippet.
      - Reads psi from gkData (varid='psi')
      - Finds X-points -> builds a 2D binary mask.
      - Returns (psiTensor, maskTensor) as a PyTorch (float) pair.
    """
    def __init__(self, paramFile, fnumList, xptCacheDir=None,
                 rotateAndReflect=False, verbosity=0):
        """
        paramFile:   Path to parameter file (string).
        fnumList:    List of frames to iterate. 
        rotateAndReflect: If True, creates static augmented copies (deprecated, use on-the-fly instead)
        """
        super().__init__()
        self.paramFile   = paramFile
        self.fnumList    = list(fnumList)  # ensure indexable
        self.xptCacheDir = xptCacheDir
        self.verbosity = verbosity

        # We'll store a base 'params' once here, and then customize in __getitem__:
        self.params = {}
        # Default snippet-based constants:
        self.params["lowerLimits"] = [-1e6, -1e6, -0.e6, -1.e6, -1.e6]
        self.params["upperLimits"] = [1e6,  1e6,   0.e6,  1.e6,  1.e6]
        self.params["restFrame"] = 1
        self.params["polyOrderOverride"] = 0

        self.params["plotContours"] = 1
        self.params["colorContours"] = 'k'
        self.params["numContours"]  = 50
        self.params["axisEqual"]    = 1
        self.params["symBar"]       = 1
        self.params["colormap"]     = 'bwr'


        # load all the data
        self.data = []
        for fnum in fnumList:
            frameData = self.load(fnum)
            self.data.append(frameData)
            if rotateAndReflect:
                self.data.append(rotate(frameData,90))
                self.data.append(rotate(frameData,180))
                self.data.append(rotate(frameData,270))
                self.data.append(reflect(frameData,0))
                self.data.append(reflect(frameData,1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load(self, fnum):
        t0 = timer()

        # check if cache exists
        if self.xptCacheDir != None:
          if not self.xptCacheDir.is_dir():
              print(f"Xpoint cache directory {self.xptCacheDir} does not exist...   exiting")
              sys.exit()
        t2 = timer()

        fields = {"psi":None,
                  "critPts":None,
                  "xpts":None,
                  "optsMax":None,
                  "optsMin":None,
                  "axesNorm":None,
                  "coords":None,
                  "fileName":None,
                  "Bx":None, "By":None,
                  "Jz":None}

        # Indicies of critical points, X points, and O points (max and min)
        if self.xptCacheDir != None and cachedPgkylDataExists(self.xptCacheDir, fnum, "psi"):
          fields = loadPgkylDataFromCache(self.xptCacheDir, fnum, fields)
        else:
          [fileName, axesNorm, critPoints, xpts, optsMax, optsMin, coords, psi, bx, by, jz] = getPgkylData(self.paramFile, fnum, verbosity=self.verbosity)
          fields = {"psi":psi, "critPts":critPoints, "xpts":xpts,
                    "optsMax":optsMax, "optsMin":optsMin,
                    "axesNorm": axesNorm, "coords": coords,
                    "fileName": fileName,
                    "Bx":bx, "By":by, "Jz":jz}
          writePgkylDataToCache(self.xptCacheDir, fnum, fields)
        self.params["axesNorm"] = fields["axesNorm"]

        if self.verbosity > 0:
          print("time (s) to find X and O points: " + str(timer()-t2))

        # Create array of 0s with 1s only at X points
        binaryMap = np.zeros(np.shape(fields["psi"]))
        binaryMap[fields["xpts"][:, 0], fields["xpts"][:, 1]] = 1

        binaryMap = expand_xpoints_mask(binaryMap, kernel_size=9)

        # Normalize input features for better training stability
        psi_norm = (fields["psi"] - fields["psi"].mean()) / (fields["psi"].std() + 1e-8)
        bx_norm = (fields["Bx"] - fields["Bx"].mean()) / (fields["Bx"].std() + 1e-8)
        by_norm = (fields["By"] - fields["By"].mean()) / (fields["By"].std() + 1e-8)
        jz_norm = (fields["Jz"] - fields["Jz"].mean()) / (fields["Jz"].std() + 1e-8)

        # -------------- 6) Convert to Torch Tensors --------------
        psi_torch = torch.from_numpy(psi_norm).float().unsqueeze(0)      # [1, Nx, Ny]
        bx_torch = torch.from_numpy(bx_norm).float().unsqueeze(0)
        by_torch = torch.from_numpy(by_norm).float().unsqueeze(0)
        jz_torch = torch.from_numpy(jz_norm).float().unsqueeze(0)
        all_torch = torch.cat((psi_torch,bx_torch,by_torch,jz_torch)) # [4, Nx, Ny]
        mask_torch = torch.from_numpy(binaryMap).float().unsqueeze(0)  # [1, Nx, Ny]

        if self.verbosity > 0:
          print("time (s) to load and process gkyl frame: " + str(timer()-t0))

        return {
            "fnum": fnum,
            "rotation": 0,
            "reflectionAxis": -1, # no reflection
            "psi": psi_torch,      # shape [1, Nx, Ny]
            "all": all_torch,      # Normalized for training
            "mask": mask_torch,    # shape [1, Nx, Ny]
            "x": fields["coords"][0],
            "y": fields["coords"][1],
            "filenameBase": fields["fileName"],
            "params": dict(self.params)  # copy of the params for local plotting
        }

class XPointPatchDataset(Dataset):
    """On‑the‑fly square crops with data augmentation, balancing positive / background patches."""
    def __init__(self, base_ds, patch=64, pos_ratio=0.5, retries=30, augment=False, seed=None):
        """
        Parameters:
        -----------
        base_ds : XPointDataset
            Base dataset containing full frames
        patch : int
            Size of square patches to extract
        pos_ratio : float
            Target ratio of patches containing X-points
        retries : int
            Number of attempts to find a suitable patch
        augment : bool
            If True, apply on-the-fly data augmentation (use for training only)
        seed : int or None
            Random seed for reproducibility (None for non-deterministic)
        """
        self.base_ds   = base_ds
        self.patch     = patch
        self.pos_ratio = pos_ratio
        self.retries   = retries
        self.augment   = augment
        
        # Initialize RNG with seed if provided
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

    def __len__(self):
        # give each full frame K random crops per epoch (K=32 for more samples)
        return len(self.base_ds) * 32

    def _crop(self, arr, top, left):
        return arr[..., top:top+self.patch, left:left+self.patch]

    def _apply_augmentation(self, all_data, mask):
        """
        Apply random data augmentation to improve generalization
        
        Augmentations applied:
        - Random rotation (90°, 180°, 270°)
        - Random horizontal flip
        - Random vertical flip
        - Gaussian noise injection
        - Random brightness/contrast adjustment
        - Cutout (random erasing)
        """
        if not self.augment:
            return all_data, mask
        
        # 1. Random rotation (0, 90, 180, 270 degrees)
        # 75% chance to apply rotation
        if self.rng.random() < 0.75:
            k = self.rng.integers(1, 4)  # 1, 2, or 3 (90°, 180°, 270°)
            all_data = torch.rot90(all_data, k=k, dims=(-2, -1))
            mask = torch.rot90(mask, k=k, dims=(-2, -1))
        
        # 2. Random horizontal flip (50% chance)
        if self.rng.random() < 0.5:
            all_data = torch.flip(all_data, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))
        
        # 3. Random vertical flip (50% chance)
        if self.rng.random() < 0.5:
            all_data = torch.flip(all_data, dims=(-2,))
            mask = torch.flip(mask, dims=(-2,))
        
        # 4. Add Gaussian noise (30% chance)
        # Small noise helps prevent overfitting to exact pixel values
        if self.rng.random() < 0.3:
            noise_std = self.rng.uniform(0.005, 0.02)
            noise = torch.randn_like(all_data) * noise_std
            all_data = all_data + noise
        
        # 5. Random brightness/contrast adjustment per channel (30% chance)
        # Helps model become invariant to intensity variations
        if self.rng.random() < 0.3:
            for c in range(all_data.shape[0]):
                brightness = self.rng.uniform(-0.1, 0.1)
                contrast = self.rng.uniform(0.9, 1.1)
                mean = all_data[c].mean()
                all_data[c] = contrast * (all_data[c] - mean) + mean + brightness
        
        # 6. Cutout/Random erasing (20% chance)
        # Prevents model from relying too heavily on specific spatial features
        if self.rng.random() < 0.2:
            h, w = all_data.shape[-2:]
            cutout_size = int(min(h, w) * self.rng.uniform(0.1, 0.25))
            if cutout_size > 0:
                y = self.rng.integers(0, max(1, h - cutout_size))
                x = self.rng.integers(0, max(1, w - cutout_size))
                all_data[..., y:y+cutout_size, x:x+cutout_size] = 0
        
        return all_data, mask

    def __getitem__(self, _):
        frame = self.base_ds[self.rng.integers(len(self.base_ds))]
        H, W  = frame["mask"].shape[-2:]

        # Ensure we have enough space for cropping
        if H < self.patch or W < self.patch:
            # Return padded version if image is too small
            all_data = F.pad(frame["all"], (0, max(0, self.patch - W), 0, max(0, self.patch - H)))
            mask = F.pad(frame["mask"], (0, max(0, self.patch - W), 0, max(0, self.patch - H)))
            
            # Apply augmentation if enabled
            all_data, mask = self._apply_augmentation(all_data, mask)
            
            return {
                "all": all_data,
                "mask": mask
            }

        for attempt in range(self.retries):
            y0 = self.rng.integers(0, H - self.patch + 1)
            x0 = self.rng.integers(0, W - self.patch + 1)
            crop_mask = self._crop(frame["mask"], y0, x0)
            has_pos   = crop_mask.sum() > 0
            want_pos  = (attempt / self.retries) < self.pos_ratio

            if has_pos == want_pos or attempt == self.retries - 1:
                all_crop = self._crop(frame["all"], y0, x0)
                
                # Apply augmentation if enabled
                all_crop, crop_mask = self._apply_augmentation(all_crop, crop_mask)
                
                return {
                    "all": all_crop,
                    "mask": crop_mask
                }


# Improved the U-Net architecture with residual connections
#   Links to understand the residual blocks:
#   https://code.likeagirl.io/u-net-vs-residual-u-net-vs-attention-u-net-vs-attention-residual-u-net-899b57c5698
#   https://notes.kvfrans.com/3-building-blocks/residual-networks.html
class ResidualBlock(nn.Module):
    """Residual block with two convolutions and skip connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection if dimensions don't match
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class UNet(nn.Module):
    """
    Improved U-Net with residual blocks and better normalization
    """
    def __init__(self, input_channels=4, base_channels=32, dropout_rate=0.2):
        super().__init__()
        
        # Encoder
        self.enc1 = ResidualBlock(input_channels, base_channels)
        self.enc2 = ResidualBlock(base_channels, base_channels*2)
        self.enc3 = ResidualBlock(base_channels*2, base_channels*4)
        self.enc4 = ResidualBlock(base_channels*4, base_channels*8)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(dropout_rate)

        # Bottleneck
        self.bottleneck = ResidualBlock(base_channels*8, base_channels*16)
        self.bottleneck_dropout = nn.Dropout2d(dropout_rate)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(base_channels*16, base_channels*8)
        self.dec4_dropout = nn.Dropout2d(dropout_rate)
        
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(base_channels*8, base_channels*4)
        self.dec3_dropout = nn.Dropout2d(dropout_rate)

        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(base_channels*4, base_channels*2)
        self.dec2_dropout = nn.Dropout2d(dropout_rate)

        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(base_channels*2, base_channels)
        self.dec1_dropout = nn.Dropout2d(dropout_rate)

        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        p2 = self.dropout(p2)
        
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        p3 = self.dropout(p3)
        
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        p4 = self.dropout(p4)

        # Bottleneck
        b = self.bottleneck(p4)
        b = self.bottleneck_dropout(b)

        # Decoder
        u4 = self.up4(b)
        cat4 = torch.cat([u4, e4], dim=1)
        d4 = self.dec4(cat4)
        d4 = self.dec4_dropout(d4)
        
        u3 = self.up3(d4)
        cat3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(cat3)
        d3 = self.dec3_dropout(d3)

        u2 = self.up2(d3)
        cat2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(cat2)
        d2 = self.dec2_dropout(d2)

        u1 = self.up1(d2)
        cat1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(cat1)
        d1 = self.dec1_dropout(d1)

        out = self.out_conv(d1)
        return out

# DICE LOSS
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, eps=1e-7):
        """
        Dice Loss implementation for binary segmentation

        Parameters:
        smooth (float): Smoothing factor to avoid division by zero, default=1.0
        eps (float): Small epsilon value to avoid numerical instability
        """
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, inputs, targets):
        """
        inputs: Model predictions (logits, before sigmoid), shape [N, 1, H, W]
        targets: Ground truth binary masks, shape [N, 1, H, W]
        """
        # Apply sigmoid to get probabilities
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate intersection and union
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()

        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth + self.eps)

        # Return Dice loss (1 - Dice coefficient)
        return 1.0 - dice

# TRAIN & VALIDATION UTILS
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp, amp_dtype, benchmark=None):
    model.train()
    running_loss = 0.0
    
    # Start epoch timing for benchmark
    if benchmark:
        benchmark.start_epoch()
    
    for batch in loader:
        batch_start = timer()
        
        all_data, mask = batch["all"].to(device), batch["mask"].to(device)
        
        with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            pred = model(all_data)
            loss = criterion(pred, mask)

        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss detected (loss = {loss.item()}). Skipping batch.")
            continue
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None: # float16 path
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        elif use_amp: # bfloat16 path (no scaler)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else: # Standard float32 path
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item()
        
        # Record batch timing for benchmark
        if benchmark:
            batch_time = timer() - batch_start
            benchmark.record_batch(all_data.size(0), batch_time)
    
    # End epoch timing for benchmark
    if benchmark:
        benchmark.end_epoch()
    
    return running_loss / len(loader) if len(loader) > 0 else 0.0

def validate_one_epoch(model, loader, criterion, device, use_amp, amp_dtype):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            all_data, mask = batch["all"].to(device), batch["mask"].to(device)
            
            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                pred = model(all_data)
                loss = criterion(pred, mask)
                
            val_loss += loss.item()
    return val_loss / len(loader) if len(loader) > 0 else 0.0

# PLOTTING FUNCTIONS
def plot_psi_contours_and_xpoints(psi_np, x, y, params, fnum, rotation,
                                  reflectionAxis, filenameBase, interpFac,
                                  xpoint_mask=None, 
                                  titleExtra="",
                                  outDir="plots", 
                                  saveFig=True):
    """
    Plots the vector potential 'psi_np' as contours, 
    then overlays X-points from xpoint_mask (if provided, shape [Nx,Ny]).
    The figure is saved to outDir
    """
    plt.figure(figsize=(12, 8))

    if params["plotContours"]:
        plt.rcParams["contour.negative_linestyle"] = "solid"
        cs = plt.contour(
            x / params["axesNorm"],
            y / params["axesNorm"],
            np.transpose(psi_np),          
            params["numContours"],
            colors=params["colorContours"],
            linewidths=0.75
        )

    plt.xlabel(r"$x/d_i$")
    plt.ylabel(r"$y/d_i$")
    if params["axisEqual"]:
        plt.gca().set_aspect("equal", "box")

    plt.title(f"Vector Potential Contours {titleExtra}, fileNum={fnum}, "
              f"reflectionAxis={reflectionAxis}")

    # Overlay X-points if xpoint_mask is given
    if xpoint_mask is not None:
        # find where xpoint_mask == 1
        xpts_row, xpts_col = np.where(xpoint_mask == 1)
        # plot as black 'x'
        plt.plot(
            x[xpts_row] / params["axesNorm"],
            y[xpts_col] / params["axesNorm"],
            'xk'
        )

    # Save the figure if needed
    if saveFig:
        basename = os.path.basename(filenameBase)
        saveFilename = os.path.join(
            outDir,
            f"{basename}_interpFac_{interpFac}_frame{fnum:04d}_rotation{rotation}_reflection{reflectionAxis}_{titleExtra.replace(' ','_')}.png"
        )
        plt.savefig(saveFilename, dpi=300)
        print("   Figure written to", saveFilename)

    plt.close()

def plot_model_performance(psi_np, pred_prob_np, mask_gt, x, y, params, fnum, filenameBase, 
                           outDir="plots", saveFig=True):
    """
    Visualize model performance comparing predictions with ground truth:
    - True Positives (green)
    - False Positives (red)
    - False Negatives (yellow)
    - Background shows psi contours
    """
    plt.figure(figsize=(12, 8))
    
    # Plot psi contours
    if params["plotContours"]:
        plt.rcParams["contour.negative_linestyle"] = "solid"
        cs = plt.contour(
            x / params["axesNorm"],
            y / params["axesNorm"],
            np.transpose(psi_np),          
            params["numContours"],
            colors='k',
            linewidths=0.75
        )
    
    # Make binary prediction
    pred_bin = (pred_prob_np[0,0] > 0.5).astype(np.float32)
    
    # Find True Positives, False Positives, False Negatives
    tp_mask = np.logical_and(pred_bin == 1, mask_gt == 1)
    fp_mask = np.logical_and(pred_bin == 1, mask_gt == 0)
    fn_mask = np.logical_and(pred_bin == 0, mask_gt == 1)
    
    # Plot each category
    tp_rows, tp_cols = np.where(tp_mask)
    fp_rows, fp_cols = np.where(fp_mask)
    fn_rows, fn_cols = np.where(fn_mask)
    
    if len(tp_rows) > 0:
        plt.plot(
            x[tp_rows] / params["axesNorm"],
            y[tp_cols] / params["axesNorm"],
            'o', color='green', markersize=8, label="True Positives"
        )
    
    if len(fp_rows) > 0:
        plt.plot(
            x[fp_rows] / params["axesNorm"],
            y[fp_cols] / params["axesNorm"],
            'o', color='red', markersize=8, label="False Positives"
        )
    
    if len(fn_rows) > 0:
        plt.plot(
            x[fn_rows] / params["axesNorm"],
            y[fn_cols] / params["axesNorm"],
            'o', color='yellow', markersize=8, label="False Negatives"
        )
    
    plt.xlabel(r"$x/d_i$")
    plt.ylabel(r"$y/d_i$")
    plt.legend(loc='best')
    
    if params["axisEqual"]:
        plt.gca().set_aspect("equal", "box")
    
    # Calculate metrics
    tp = np.sum(tp_mask)
    fp = np.sum(fp_mask)
    fn = np.sum(fn_mask)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    plt.title(f"Model Performance, fileNum={fnum}\nPrecision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    if saveFig:
        basename = os.path.basename(filenameBase)
        saveFilename = os.path.join(
            outDir,
            f"{basename}_model_performance_{fnum:04d}.png"
        )
        plt.savefig(saveFilename, dpi=300)
        print("   Model performance figure written to", saveFilename)
    
    plt.close()

def plot_training_history(train_losses, val_loss, save_path='plots/training_history.png'):
    """
    Plots training and validation losses across epochs.
    
    Parameters:
    train_losses (list): List of training losses for each epoch
    val_loss (list): List of validation losses for each epoch
    save_path (str): Path to save the resulting plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add some padding to y-axis to make visualization clearer
    ymin = min(min(train_losses), min(val_loss)) * 0.9
    ymax = max(max(train_losses), max(val_loss)) * 1.1
    plt.ylim(ymin, ymax)
    
    plt.savefig(save_path, dpi=300)
    print(f"Training history plot saved to {save_path}")
    plt.close()

def parseCommandLineArgs():
    parser = argparse.ArgumentParser(description='ML-based reconnection classifier')
    parser.add_argument('--learningRate', type=float, default=1e-5,
                        help='specify the learning rate')
    parser.add_argument('--weightDecay', type=float, default=1e-4,
                        help='specify the weight decay (L2 regularization) for optimizer')
    parser.add_argument('--dropoutRate', type=float, default=0.2,
                        help='specify the dropout rate for regularization')
    parser.add_argument('--batchSize', type=int, default=1,
                        help='specify the batch size')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='specify the number of epochs')
    parser.add_argument('--trainFrameFirst', type=int, default=1,
                        help='specify the number of the first frame used for training')
    parser.add_argument('--trainFrameLast', type=int, default=140,
                        help='specify the number of the last frame (exclusive) used for training')
    parser.add_argument('--validationFrameFirst', type=int, default=141,
                        help='specify the number of the first frame used for validation')
    parser.add_argument('--validationFrameLast', type=int, default=150,
                        help='specify the number of the last frame (exclusive) used for validation')
    parser.add_argument('--minTrainingLoss', type=int, default=3,
                        help='''
                        minimum reduction in training loss in orders of magnitude,
                        set to 0 to disable the check (default: 3)
                        ''')
    parser.add_argument('--checkPointFrequency', type=int, default=10,
                        help='number of epochs between checkpoints')
    parser.add_argument('--paramFile', type=Path, default=None,
                        help='''
                        specify the path to the parameter txt file, the parent
                        directory of that file must contain the gkyl input training data
                        ''')
    parser.add_argument('--xptCacheDir', type=Path, default=None,
                        help='''
                        specify the path to a directory that will be used to cache
                        the outputs of the analytic Xpoint finder
                        ''')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction,
                        help='create figures of the ground truth X-points and model identified X-points')
    parser.add_argument('--plotDir', type=Path, default="./plots",
                        help='directory where figures are written')
    parser.add_argument('--use-amp', action='store_true',
                        help='use automatic mixed precision training')
    parser.add_argument('--amp-dtype', type=str, default='bfloat16', 
                        choices=['float16', 'bfloat16'], help='data type for mixed precision (bfloat16 recommended)')
    parser.add_argument('--patience', type=int, default=15,
                        help='patience for early stopping (default: 15)')
    parser.add_argument('--benchmark', action='store_true',
                        help='enable performance benchmarking (tracks timing, throughput, GPU memory)')
    parser.add_argument('--benchmark-output', type=Path, default='./benchmark_results.json',
                        help='path to save benchmark results JSON file (default: ./benchmark_results.json)')
    parser.add_argument('--eval-output', type=Path, default='./evaluation_metrics.json',
                        help='path to save evaluation metrics JSON file (default: ./evaluation_metrics.json)')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed for reproducibility (default: None for non-deterministic)')
    parser.add_argument('--require-gpu', action='store_true',
                        help='require GPU to be available, exit if not found')
    
    # CI TEST: Add smoke test flag
    parser.add_argument('--smoke-test', action='store_true',
                        help='Run a minimal smoke test for CI (overrides other parameters)')
    
    args = parser.parse_args()
    return args

def checkCommandLineArgs(args):
    # CI TEST: Skip file checks in smoke test mode
    if args.smoke_test:
        return
        
    if args.xptCacheDir != None:
      if not args.xptCacheDir.is_dir():
          print(f"Xpoint cache directory {args.xptCacheDir} does not exist. "
                 "Please create the directory... exiting")
          sys.exit()

    if args.paramFile == None:
      print(f"parameter file required but not set... exiting")
      sys.exit()
    if args.paramFile.is_dir():
      print(f"parameter file {args.paramFile} is a directory ... exiting")
      sys.exit()
    if not args.paramFile.exists():
      print(f"parameter file {args.paramFile} does not exist... exiting")
      sys.exit()

    if args.trainFrameFirst == 0 or args.validationFrameFirst == 0:
      print(f"frame 0 does not contain valid data... exiting")
      sys.exit()

    if args.trainFrameLast <= args.trainFrameFirst:
      print(f"training frame range isn't valid... exiting")
      sys.exit()

    if args.validationFrameLast <= args.validationFrameFirst:
      print(f"validation frame range isn't valid... exiting")
      sys.exit()

    if args.learningRate <= 0:
      print(f"learningRate must be > 0... exiting")
      sys.exit()

    if args.batchSize < 1:
      print(f"batchSize must be >= 1... exiting")
      sys.exit()

    if args.minTrainingLoss < 0:
      print(f"minTrainingLoss must be >= 0... exiting")
      sys.exit()

    if args.checkPointFrequency < 0:
      print(f"checkPointFrequency must be >= 0... exiting")
      sys.exit()

def printCommandLineArgs(args):
    print("Config {")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("}")

# Function to save model checkpoint
def save_model_checkpoint(model, optimizer, train_loss, val_loss, epoch, checkpoint_dir="checkpoints", scaler=None, best_val_loss=None):
    """
    Save model checkpoint including model state, optimizer state, and training metrics
    
    Parameters:
    model: The neural network model
    optimizer: The optimizer used for training
    train_loss: List of training losses
    val_loss: List of validation losses
    epoch: Current epoch number
    checkpoint_dir: Directory to save checkpoints
    scaler: GradScaler instance if using AMP
    best_val_loss: Best validation loss so far
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"xpoint_model_epoch_{epoch}.pt")
    
    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss
    }
    
    # Save scaler state if using AMP
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved at epoch {epoch} to {checkpoint_path}")

    # create symbolic link to latest checkpoint
    target_path = f"xpoint_model_epoch_{epoch}.pt"
    latest_path = os.path.join(checkpoint_dir, "xpoint_model_latest.pt")
    try:
        os.symlink(target_path, latest_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(latest_path)
            os.symlink(target_path, latest_path)
        else:
            raise e

# Function to load model checkpoint
def load_model_checkpoint(model, optimizer, checkpoint_path, scaler=None):
    """
    Load model checkpoint
    
    Parameters:
    model: The neural network model to load weights into
    optimizer: The optimizer to load state into
    checkpoint_path: Path to the checkpoint file
    scaler: GradScaler instance if using AMP
    
    Returns:
    model: Updated model with loaded weights
    optimizer: Updated optimizer with loaded state
    epoch: Last saved epoch number
    train_loss: List of training losses
    val_loss: List of validation losses
    scaler: Updated scaler if using AMP
    best_val_loss: Best validation loss from checkpoint
    """
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return model, optimizer, 0, [], [], scaler, float('inf')
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)  # Need False for optimizer state
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Load scaler state if available
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    print(f"Loaded checkpoint from epoch {epoch}")
    return model, optimizer, epoch, train_loss, val_loss, scaler, best_val_loss


def main():
    args = parseCommandLineArgs()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # CI TEST: Override parameters for smoke test
    if args.smoke_test:
        print("=" * 60)
        print("RUNNING IN SMOKE TEST MODE FOR CI")
        print("=" * 60)
        
        # Override with minimal parameters
        args.epochs = 5
        args.batchSize = 1
        args.trainFrameFirst = 1
        args.trainFrameLast = 11   # 10 frames for training
        args.validationFrameFirst = 11
        args.validationFrameLast = 12  # 1 frame for validation
        args.plot = False  # Disable plotting for CI
        args.checkPointFrequency = 2  # Save more frequently
        args.minTrainingLoss = 0  # Don't fail on convergence in smoke test
        
        print("Smoke test parameters:")
        print(f"  - Training frames: {args.trainFrameFirst} to {args.trainFrameLast-1}")
        print(f"  - Validation frames: {args.validationFrameFirst} to {args.validationFrameLast-1}")
        print(f"  - Epochs: {args.epochs}")
        print(f"  - Batch size: {args.batchSize}")
        print(f"  - Plotting disabled")
        print("=" * 60)
    
    checkCommandLineArgs(args)
    printCommandLineArgs(args)

    # output directory:
    outDir = args.plotDir
    os.makedirs(outDir, exist_ok=True)

    t0 = timer()
    
    # CI TEST: Use synthetic data for smoke test
    if args.smoke_test:
        print("\nUsing synthetic data for smoke test...")
        train_dataset = SyntheticXPointDataset(nframes=10, shape=(64, 64), nxpoints=3)
        val_dataset = SyntheticXPointDataset(nframes=1, shape=(64, 64), nxpoints=3, seed=123)
        print(f"Created synthetic datasets: {len(train_dataset)} train, {len(val_dataset)} val frames")
    else:
        # Original data loading - NO STATIC AUGMENTATION
        train_fnums = range(args.trainFrameFirst, args.trainFrameLast)
        val_fnums   = range(args.validationFrameFirst, args.validationFrameLast)
        
        print(f"Loading training data from frames {args.trainFrameFirst} to {args.trainFrameLast-1}")
        print(f"Loading validation data from frames {args.validationFrameFirst} to {args.validationFrameLast-1}")
        
        # Set rotateAndReflect=False - we'll use on-the-fly augmentation instead
        train_dataset = XPointDataset(args.paramFile, train_fnums,
            xptCacheDir=args.xptCacheDir, rotateAndReflect=False)
        val_dataset   = XPointDataset(args.paramFile, val_fnums,
            xptCacheDir=args.xptCacheDir, rotateAndReflect=False)
    
    # Enable augmentation for training, disable for validation
    train_crop = XPointPatchDataset(train_dataset, patch=64, pos_ratio=0.5, retries=30, 
                                    augment=True, seed=args.seed)
    val_crop   = XPointPatchDataset(val_dataset, patch=64, pos_ratio=0.5, retries=30, 
                                    augment=False, seed=args.seed)

    t1 = timer()
    print("time (s) to create gkyl data loader: " + str(t1-t0))
    print(f"number of training frames: {len(train_dataset)}")
    print(f"number of validation frames: {len(val_dataset)}")
    print(f"number of training patches per epoch: {len(train_crop)}")
    print(f"number of validation patches per epoch: {len(val_crop)}")
    print(f"Data augmentation: ENABLED for training, DISABLED for validation")
    if args.seed is not None:
        print(f"Random seed: {args.seed} (reproducible mode)")
    else:
        print(f"Random seed: None (non-deterministic mode)")

    train_loader = DataLoader(train_crop, batch_size=args.batchSize, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_crop,   batch_size=args.batchSize, shuffle=False, num_workers=0)

    # Check GPU requirement
    if args.require_gpu and not torch.cuda.is_available():
        print("=" * 60)
        print("ERROR: GPU required but not available!")
        print("=" * 60)
        print("The --require-gpu flag was set, but CUDA is not available.")
        print("Please check:")
        print("  1. NVIDIA GPU is properly installed")
        print("  2. CUDA drivers are installed")
        print("  3. PyTorch was installed with CUDA support")
        print("\nExiting...")
        sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.require_gpu:
        print("GPU requirement: ENABLED (will exit if GPU not available)")
    
    # Initialize benchmark tracker
    benchmark = TrainingBenchmark(device, enabled=args.benchmark)
    if args.benchmark:
        benchmark.print_hardware_info()
    
    # Use the improved model
    model = UNet(input_channels=4, base_channels=32, dropout_rate=args.dropoutRate).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Dropout rate: {args.dropoutRate}")

    criterion = DiceLoss(smooth=1.0)
    
    # Use AdamW optimizer with weight decay for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=args.learningRate, weight_decay=args.weightDecay)
    print(f"Optimizer: AdamW with learning_rate={args.learningRate}, weight_decay={args.weightDecay}")
    
    # Learning rate scheduler with cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # --- AMP Setup (bfloat16 aware) ---
    use_amp = args.use_amp and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' and torch.cuda.is_bf16_supported() else torch.float16
    
    # GradScaler is ONLY needed for float16, not bfloat16
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    if use_amp:
        if args.amp_dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
            print("Warning: bfloat16 not supported on this GPU. Falling back to float16.")
        print(f"Using Automatic Mixed Precision (AMP) with dtype: {amp_dtype}")

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint_path = os.path.join(checkpoint_dir, "xpoint_model_latest.pt")
    start_epoch = 0
    train_loss = []
    val_loss = []
    best_val_loss = float('inf')

    if os.path.exists(latest_checkpoint_path) and not args.smoke_test:
        model, optimizer, start_epoch, train_loss, val_loss = load_model_checkpoint(
            model, optimizer, latest_checkpoint_path
        )
        print(f"Resuming training from epoch {start_epoch+1}")
        print(f"Best validation loss so far: {best_val_loss:.6f}")
    else:
        print("Starting training from scratch")

    t2 = timer()
    print("time (s) to prepare model: " + str(t2-t1))

    # Early stopping setup
    patience_counter = 0
    
    num_epochs = args.epochs
    for epoch in range(start_epoch, num_epochs):
        train_loss_epoch = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp, amp_dtype, benchmark)
        val_loss_epoch = validate_one_epoch(model, val_loader, criterion, device, use_amp, amp_dtype)
        
        train_loss.append(train_loss_epoch)
        val_loss.append(val_loss_epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Enhanced logging with benchmark metrics
        log_msg = f"[Epoch {epoch+1}/{num_epochs}] LR={current_lr:.2e} TrainLoss={train_loss[-1]:.6f} ValLoss={val_loss[-1]:.6f}"
        if args.benchmark:
            throughput = benchmark.get_throughput()
            gpu_mem = benchmark.get_gpu_memory_usage()
            log_msg += f" | Throughput={throughput:.2f} samples/s | GPU Mem={gpu_mem:.2f} GB"
        print(log_msg)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Check for improvement
        if val_loss[-1] < best_val_loss:
            best_val_loss = val_loss[-1]
            patience_counter = 0
            print(f"   New best validation loss: {best_val_loss:.6f}")
            # Save best model
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))
        else:
            patience_counter += 1
            
        # Save checkpoint periodically
        if (epoch+1) % args.checkPointFrequency == 0:
            save_model_checkpoint(model, optimizer, train_loss, val_loss, epoch+1, checkpoint_dir, scaler, best_val_loss)
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs (patience={args.patience})")
            break

    plot_training_history(train_loss, val_loss, save_path='plots/training_history.png')
    print("time (s) to train model: " + str(timer()-t2))
    
    # Print and save benchmark summary
    if args.benchmark:
        benchmark.print_summary(output_file=args.benchmark_output)

    # CI TEST: Run additional tests if in smoke test mode
    if args.smoke_test:
        print("\n" + "="*60)
        print("SMOKE TEST: Running additional CI tests")
        print("="*60)
        
        # Test 1: Checkpoint save/load
        checkpoint_test_passed = test_checkpoint_functionality(
            model, optimizer, device, val_loader, criterion, None, UNet, optim.Adam
        )
        
        # Test 2: Inference test
        print("Running inference test...")
        model.eval()
        with torch.no_grad():
            # Get one batch
            test_batch = next(iter(val_loader))
            test_input = test_batch["all"].to(device)
            test_output = model(test_input)
            
            # Apply sigmoid to get probabilities
            test_probs = torch.sigmoid(test_output)
            
            print(f"Input shape: {test_input.shape}")
            print(f"Output shape: {test_output.shape}")
            print(f"Output range (logits): [{test_output.min():.3f}, {test_output.max():.3f}]")
            print(f"Output range (probs): [{test_probs.min():.3f}, {test_probs.max():.3f}]")
            print(f"Predicted X-points: {(test_probs > 0.5).sum().item()} pixels")
        
        # Test 3: Check if model learned anything
        initial_train_loss = train_loss[0] if train_loss else float('inf')
        final_train_loss = train_loss[-1] if train_loss else float('inf')
        
        print(f"\nTraining progress:")
        print(f"Initial loss: {initial_train_loss:.6f}")
        print(f"Final loss: {final_train_loss:.6f}")
        
        if final_train_loss < initial_train_loss:
            print("✓ Model showed improvement during training")
            training_improved = True
        else:
            print("✗ Model did not improve during training")
            training_improved = False
        
        # Overall smoke test result
        print("\n" + "="*60)
        print("SMOKE TEST SUMMARY")
        print("="*60)
        print(f"Checkpoint test: {'PASSED' if checkpoint_test_passed else 'FAILED'}")
        print(f"Training improvement: {'YES' if training_improved else 'NO'}")
        print(f"Overall result: {'PASSED' if checkpoint_test_passed else 'FAILED'}")
        print("="*60)
        
        # Return appropriate exit code for CI
        if not checkpoint_test_passed:
            return 1
        else:
            return 0

    # Check training progress
    if len(train_loss) > 1 and train_loss[-1] > 0 and train_loss[0] > 0:
        loss_reduction = np.log10(abs(train_loss[0]/train_loss[-1]))
        print(f"Training loss reduced by {loss_reduction:.2f} orders of magnitude")
        if args.minTrainingLoss > 0 and loss_reduction < args.minTrainingLoss:
            print(f"Warning: TrainLoss reduced by less than {args.minTrainingLoss} orders of magnitude")

    # Load best model for evaluation
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        print("Loading best model for evaluation...")
        model.load_state_dict(torch.load(best_model_path, weights_only=True))

    # new evaluation code
    # Evaluate model performance
    if not args.smoke_test:
        # print("\n" + "="*70)
        # print("RUNNING MODEL EVALUATION")
        # print("="*70)
        
        # # Evaluate on validation set
        # print("\nEvaluating on validation set...")
        val_evaluator = evaluate_model_on_dataset(
            model, 
            val_dataset,  # Use original dataset, not patch dataset
            device, 
            use_amp=use_amp, 
            amp_dtype=amp_dtype,
            threshold=0.5
        )
        
        # Print and save validation metrics
        val_evaluator.print_summary()
        val_evaluator.save_json(args.eval_output)
        
        # Evaluate on training set
        print("\nEvaluating on training set...")
        train_evaluator = evaluate_model_on_dataset(
            model, 
            train_dataset,
            device, 
            use_amp=use_amp, 
            amp_dtype=amp_dtype,
            threshold=0.5
        )
        
        # Print and save training metrics
        train_evaluator.print_summary()
        train_eval_path = args.eval_output.parent / f"train_{args.eval_output.name}"
        train_evaluator.save_json(train_eval_path)
        
        # Compare training vs validation to check for overfitting
        train_global = train_evaluator.get_global_metrics()
        val_global = val_evaluator.get_global_metrics()
        
        print("\n" + "="*70)
        print("OVERFITTING CHECK")
        print("="*70)
        print(f"Training F1:      {train_global['f1_score']:.4f}")
        print(f"Validation F1:    {val_global['f1_score']:.4f}")
        print(f"Difference:       {abs(train_global['f1_score'] - val_global['f1_score']):.4f}")
        
        if train_global['f1_score'] - val_global['f1_score'] > 0.05:
            print("⚠ Warning: Possible overfitting detected (train F1 >> val F1)")
        elif val_global['f1_score'] - train_global['f1_score'] > 0.05:
            print("⚠ Warning: Unusual pattern (val F1 >> train F1)")
        else:
            print("✓ Model generalizes well to validation set")
        print("="*70 + "\n")
    
    # ==================== END NEW EVALUATION CODE ====================

    # (D) Plotting after training
    model.eval() # switch to inference mode
    outDir = "plots"
    interpFac = 1  

    # Evaluate on combined set for demonstration
    if not args.smoke_test:
        train_fnums = range(args.trainFrameFirst, args.trainFrameLast)
        val_fnums   = range(args.validationFrameFirst, args.validationFrameLast)
        full_dataset = [train_dataset, val_dataset]
    else:
        full_dataset = [val_dataset]  # Only use validation data for smoke test

    t4 = timer()

    with torch.no_grad():
        for dataset in full_dataset:  
            for item in dataset:  
                fnum     = item["fnum"]
                rotation = item["rotation"]
                reflectionAxis = item["reflectionAxis"]
                psi_np   = np.array(item["psi"])[0]
                mask_gt  = np.array(item["mask"])[0]
                x        = item["x"]
                y        = item["y"]
                filenameBase      = item["filenameBase"]
                params   = item["params"]

                # Get CNN prediction
                all_torch = item["all"].unsqueeze(0).to(device)
                
                with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    pred_mask = model(all_torch)
                    pred_prob = torch.sigmoid(pred_mask)
                
                # Convert to float32 before numpy conversion (fixes BFloat16 error)
                pred_mask_np = pred_mask[0,0].float().cpu().numpy()
                pred_prob_np = pred_prob.float().cpu().numpy()

                pred_mask_bin = (pred_prob_np[0,0] > 0.5).astype(np.float32)

                if args.plot:
                    # Plot GROUND TRUTH
                    plot_psi_contours_and_xpoints(
                        psi_np, x, y, params, fnum, rotation, reflectionAxis, filenameBase, interpFac,
                        xpoint_mask=mask_gt,
                        titleExtra="GTXpoints",
                        outDir=outDir,
                        saveFig=True
                    )

                    # Plot CNN PREDICTIONS
                    plot_psi_contours_and_xpoints(
                        psi_np, x, y, params, fnum, rotation, reflectionAxis, filenameBase, interpFac,
                        xpoint_mask=pred_mask_bin,
                        titleExtra="CNNXpoints",
                        outDir=outDir,
                        saveFig=True
                    )

                    plot_model_performance(
                        psi_np, pred_prob_np, mask_gt, x, y, params, fnum, filenameBase,
                        outDir=outDir,
                        saveFig=True
                    )

    t5 = timer()
    print("time (s) to apply model: " + str(t5-t4))
    print("total time (s): " + str(t5-t0))

if __name__ == "__main__":
    main()