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
    all_data = v2.functional.rotate(frameData["all"], deg, v2.InterpolationMode.BILINEAR)
    mask = v2.functional.rotate(frameData["mask"], deg, v2.InterpolationMode.BILINEAR)
    return {
        "fnum": frameData["fnum"],
        "rotation": deg,
        "reflectionAxis": -1, # no reflection
        "psi": psi,
        "all": all_data,
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
    psi = torch.flip(frameData["psi"][0], dims=(axis,)).unsqueeze(0)
    all_data = torch.flip(frameData["all"], dims=(axis,))
    mask = torch.flip(frameData["mask"][0], dims=(axis,)).unsqueeze(0)
    return {
        "fnum": frameData["fnum"],
        "rotation": 0,
        "reflectionAxis": axis,
        "psi": psi,
        "all": all_data,
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
    if cacheDir is None:
        return False
    else:
        cachedFrame = cacheDir / f"{frameNumber}_{fieldName}.npy"
        return cachedFrame.exists()

def loadPgkylDataFromCache(cacheDir, frameNumber, fields):
    outFields = {}
    if cacheDir is not None:
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
    if cacheDir is not None:
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
        if self.xptCacheDir is not None:
            if not self.xptCacheDir.is_dir():
                print(f"Xpoint cache directory {self.xptCacheDir} does not exist... exiting")
                sys.exit()
        
        t2 = timer()
        fields = {"psi":None, "critPts":None, "xpts":None, "optsMax":None, "optsMin":None, "axesNorm":None, "coords":None, "fileName":None, "Bx":None, "By":None, "Jz":None}

        # Indicies of critical points, X points, and O points (max and min)
        if self.xptCacheDir is not None and cachedPgkylDataExists(self.xptCacheDir, fnum, "psi"):
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

        # -------------- 6) Convert to Torch Tensors --------------
        psi_torch = torch.from_numpy(fields["psi"]).float().unsqueeze(0)    # [1, Nx, Ny]
        bx_torch = torch.from_numpy(fields["Bx"]).float().unsqueeze(0)
        by_torch = torch.from_numpy(fields["By"]).float().unsqueeze(0)
        jz_torch = torch.from_numpy(fields["Jz"]).float().unsqueeze(0)
        
        all_torch = torch.cat((psi_torch,bx_torch,by_torch,jz_torch)) # [4, Nx, Ny]
        mask_torch = torch.from_numpy(binaryMap).float().unsqueeze(0)  # [1, Nx, Ny]

        if self.verbosity > 0:
            print("time (s) to load and process gkyl frame: " + str(timer()-t0))

        return {
            "fnum": fnum,
            "rotation": 0,
            "reflectionAxis": -1, # no reflection
            "psi": psi_torch,       # shape [1, Nx, Ny]
            "all": all_torch,       # shape [4, Nx, Ny]
            "mask": mask_torch,     # shape [1, Nx, Ny]
            "x": fields["coords"][0],
            "y": fields["coords"][1],
            "filenameBase": fields["fileName"],
            "params": dict(self.params)  # copy of the params for local plotting
        }

class XPointPatchDataset(Dataset):
    """On‑the‑fly square crops, balancing positive / background patches."""
    def __init__(self, base_ds, patch=64, pos_ratio=0.6, retries=20):
        self.base_ds   = base_ds
        self.patch     = patch
        self.pos_ratio = pos_ratio
        self.retries   = retries
        self.rng       = np.random.default_rng()
        # Precompute some statistics for normalization
        self.compute_normalization_stats()
    
    def compute_normalization_stats(self):
        """Compute global mean and std for normalization"""
        # Sample a few frames to compute statistics
        n_samples = min(10, len(self.base_ds))
        all_values = []
        
        for i in range(n_samples):
            frame = self.base_ds[i]
            all_values.append(frame["all"].numpy())
        
        all_values = np.concatenate([v.flatten() for v in all_values])
        self.global_mean = np.mean(all_values)
        self.global_std = np.std(all_values)
        
        # Prevent division by zero
        if self.global_std == 0:
            self.global_std = 1.0
            
        print(f"Computed normalization stats: mean={self.global_mean:.4f}, std={self.global_std:.4f}")

    def __len__(self):
        # give each full frame K random crops per epoch (K=16 by default)
        return len(self.base_ds) * 16

    def _crop(self, arr, top, left):
        return arr[..., top:top+self.patch, left:left+self.patch]

    def __getitem__(self, _):
        frame = self.base_ds[self.rng.integers(len(self.base_ds))]
        H, W  = frame["mask"].shape[-2:]

        # comments on the logic
        for attempt in range(self.retries):
            y0 = self.rng.integers(0, H - self.patch + 1)
            x0 = self.rng.integers(0, W - self.patch + 1)
            crop_mask = self._crop(frame["mask"], y0, x0)
            has_pos   = crop_mask.sum() > 0
            want_pos  = (attempt / self.retries) < self.pos_ratio

            if has_pos == want_pos or attempt == self.retries - 1:
                crop_all = self._crop(frame["all"],  y0, x0)
                # Apply global normalization
                crop_all = (crop_all - self.global_mean) / self.global_std
                
                return {
                    "all" : crop_all,
                    "mask": crop_mask
                }

# 2) U-NET ARCHITECTURE
class UNet(nn.Module):
    """
    A simplified U-Net for binary segmentation:
      in:  (N, 1,   H, W)   ++++ BX, BY, JZ
      out: (N, 1,   H, W)
    """
    def __init__(self, input_channels=4, base_channels=64):
        super().__init__()
        self.enc1 = self.double_conv(input_channels, base_channels)      # -> base_channels
        self.enc2 = self.double_conv(base_channels, base_channels*2)    # -> 2*base_channels
        self.enc3 = self.double_conv(base_channels*2, base_channels*4)  # -> 4*base_channels
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self.double_conv(base_channels*4, base_channels*8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.dec3 = self.double_conv(base_channels*8, base_channels*4)

        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.dec2 = self.double_conv(base_channels*4, base_channels*2)

        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec1 = self.double_conv(base_channels*2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1)
        
        # Initialize weights for better stability
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                # shape: [N, base_channels, H, W]
        p1 = self.pool(e1)               # half spatial dims

        e2 = self.enc2(p1)               # [N, 2*base_channels, H/2, W/2]
        p2 = self.pool(e2)

        e3 = self.enc3(p2)               # [N, 4*base_channels, H/4, W/4]
        p3 = self.pool(e3)               # [N, 4*base_channels, H/8, W/8]

        # Bottleneck
        b  = self.bottleneck(p3)         # [N, 8*base_channels, H/8, W/8]

        # Decoder
        u3 = self.up3(b)                 # -> shape ~ e3
        cat3 = torch.cat([u3, e3], dim=1)  # skip connection
        d3 = self.dec3(cat3)

        u2 = self.up2(d3)                # -> shape ~ e2
        cat2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(cat2)

        u1 = self.up1(d2)                # -> shape ~ e1
        cat1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(cat1)

        out = self.out_conv(d1)
        return out  # We'll apply sigmoid in the loss or after

# TRAIN & VALIDATION UTILS
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, use_amp=False, amp_dtype=torch.float16):
    model.train()
    running_loss = 0.0
    num_batches = 0
    num_skipped = 0
    
    for batch_idx, batch in enumerate(loader):
        all_data, mask = batch["all"].to(device), batch["mask"].to(device)
        
        if use_amp:
            # Clear gradients
            optimizer.zero_grad()
            
            # Use autocast for forward pass
            with autocast(device_type='cuda', dtype=amp_dtype):
                pred = model(all_data)
                loss = criterion(pred, mask)
            
            # Check if loss is valid
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss detected in batch {batch_idx}, skipping...")
                num_skipped += 1
                continue
            
            # For bfloat16, we don't use GradScaler
            if amp_dtype == torch.bfloat16:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if not torch.isfinite(grad_norm):
                    print(f"Warning: Non-finite gradients detected in batch {batch_idx}, skipping...")
                    num_skipped += 1
                    optimizer.zero_grad()
                    continue
                
                optimizer.step()
            else:
                # Use GradScaler for float16
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
                
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                
                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Check gradient norm
                if not torch.isfinite(grad_norm):
                    print(f"Warning: Non-finite gradients detected in batch {batch_idx}, skipping...")
                    num_skipped += 1
                    optimizer.zero_grad()  # Clear the invalid gradients
                    scaler.update()  # Update scaler state
                    continue
                
                # Optimizer step and scaler update
                scaler.step(optimizer)
                scaler.update()
            
        else:
            # Standard training without AMP
            optimizer.zero_grad()
            pred = model(all_data)
            loss = criterion(pred, mask)
            
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss detected in batch {batch_idx}, skipping...")
                num_skipped += 1
                continue
            
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if not torch.isfinite(grad_norm):
                print(f"Warning: Non-finite gradients detected in batch {batch_idx}, skipping...")
                num_skipped += 1
                optimizer.zero_grad()
                continue
            
            optimizer.step()
        
        running_loss += loss.item()
        num_batches += 1
    
    if num_skipped > 0:
        print(f"  Skipped {num_skipped} batches due to numerical issues")
    
    return running_loss / max(num_batches, 1)

def validate_one_epoch(model, loader, criterion, device, use_amp=False, amp_dtype=torch.float16):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            all_data, mask = batch["all"].to(device), batch["mask"].to(device)
            
            if use_amp:
                with autocast(device_type='cuda', dtype=amp_dtype):
                    pred = model(all_data)
                    loss = criterion(pred, mask)
            else:
                pred = model(all_data)
                loss = criterion(pred, mask)
                
            val_loss += loss.item()
    return val_loss / len(loader)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss implementation
        
        Parameters:
        alpha (float): Weighting factor for the rare class (X-points), default=0.25
        gamma (float): Focusing parameter that reduces the loss for well-classified examples, default=2.0
        reduction (str): 'mean' or 'sum', how to reduce the loss over the batch
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        inputs: Model predictions (logits, before sigmoid), shape [N, 1, H, W]
        targets: Ground truth binary masks, shape [N, 1, H, W]
        """
        # Apply sigmoid to get probabilities
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Get probabilities for positive class
        probs = torch.sigmoid(inputs)
        # For targets=1 (X-points), pt = p; for targets=0 (non-X-points), pt = 1-p
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Apply focusing parameter
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting: alpha for X-points, (1-alpha) for non-X-points
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Combine all factors
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

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
        
        # Ensure inputs are in valid range to prevent NaN
        inputs = torch.clamp(inputs, min=self.eps, max=1.0 - self.eps)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate intersection and union
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()

        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth + self.eps)

        # Return Dice loss (1 - Dice coefficient)
        return 1.0 - dice

# PLOTTING FUNCTION
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
    Visualize model performance comparing predictions with ground truth.
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
        plt.plot(x[tp_rows] / params["axesNorm"], y[tp_cols] / params["axesNorm"],
                 'o', color='green', markersize=8, label="True Positives")
    
    if len(fp_rows) > 0:
        plt.plot(x[fp_rows] / params["axesNorm"], y[fp_cols] / params["axesNorm"],
                 'o', color='red', markersize=8, label="False Positives")
    
    if len(fn_rows) > 0:
        plt.plot(x[fn_rows] / params["axesNorm"], y[fn_cols] / params["axesNorm"],
                 'o', color='yellow', markersize=8, label="False Negatives")
    
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

def plot_training_history(train_losses, val_losses, save_path='plots/training_history.png'):
    """
    Plots training and validation losses across epochs.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    # Filter out NaN values for plotting
    train_losses_clean = [loss if not np.isnan(loss) else None for loss in train_losses]
    val_losses_clean = [loss if not np.isnan(loss) else None for loss in val_losses]
    
    plt.plot(epochs, train_losses_clean, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses_clean, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add some padding to y-axis to make visualization clearer
    # Handle case where all values might be NaN
    valid_losses = [loss for loss in train_losses + val_losses if loss is not None and not np.isnan(loss)]
    if valid_losses:
        ymin = min(valid_losses) * 0.9
        ymax = max(valid_losses) * 1.1
        plt.ylim(ymin, ymax)
    
    plt.savefig(save_path, dpi=300)
    print(f"Training history plot saved to {save_path}")
    plt.close()

def parseCommandLineArgs():
    parser = argparse.ArgumentParser(description='ML-based reconnection classifier')
    parser.add_argument('--learningRate', type=float, default=1e-5, help='specify the learning rate')
    parser.add_argument('--batchSize', type=int, default=1, help='specify the batch size')
    parser.add_argument('--epochs', type=int, default=2000, help='specify the number of epochs')
    parser.add_argument('--trainFrameFirst', type=int, default=1, help='specify the number of the first frame used for training')
    parser.add_argument('--trainFrameLast', type=int, default=140, help='specify the number of the last frame (exclusive) used for training')
    parser.add_argument('--validationFrameFirst', type=int, default=141, help='specify the number of the first frame used for validation')
    parser.add_argument('--validationFrameLast', type=int, default=150, help='specify the number of the last frame (exclusive) used for validation')
    parser.add_argument('--minTrainingLoss', type=int, default=3, help='''minimum reduction in training loss in orders of magnitude, set to 0 to disable the check''')
    parser.add_argument('--checkPointFrequency', type=int, default=10, help='number of epochs between checkpoints')
    parser.add_argument('--paramFile', type=Path, default=None, help='''specify the path to the parameter txt file, the parent directory of that file must contain the gkyl input training data''')
    parser.add_argument('--xptCacheDir', type=Path, default=None, help='''specify the path to a directory that will be used to cache the outputs of the analytic Xpoint finder''')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, help='create figures of the ground truth X-points and model identified X-points')
    parser.add_argument('--plotDir', type=Path, default="./plots", help='directory where figures are written')
    parser.add_argument('--use-amp', action='store_true', help='use automatic mixed precision training')
    parser.add_argument('--amp-dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type for mixed precision (float16 or bfloat16)')
    args = parser.parse_args()
    return args

def checkCommandLineArgs(args):
    if args.xptCacheDir is not None:
        if not args.xptCacheDir.is_dir():
            print(f"Xpoint cache directory {args.xptCacheDir} does not exist. Please create the directory... exiting")
            sys.exit()

    if args.paramFile is None:
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
def save_model_checkpoint(model, optimizer, train_loss, val_loss, epoch, checkpoint_dir="checkpoints", scaler=None):
    """
    Save model checkpoint including model state, optimizer state, and training metrics
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"xpoint_model_epoch_{epoch}.pt")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
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
    """
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return model, optimizer, 0, [], [], scaler
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    print(f"Loaded checkpoint from epoch {epoch}")
    return model, optimizer, epoch, train_loss, val_loss, scaler

def main():
    args = parseCommandLineArgs()
    checkCommandLineArgs(args)
    printCommandLineArgs(args)

    # output directory:
    outDir = args.plotDir
    os.makedirs(outDir, exist_ok=True)

    t0 = timer()
    train_fnums = range(args.trainFrameFirst, args.trainFrameLast)
    val_fnums   = range(args.validationFrameFirst, args.validationFrameLast)

    train_dataset = XPointDataset(args.paramFile, train_fnums,
            xptCacheDir=args.xptCacheDir, rotateAndReflect=True)
    val_dataset   = XPointDataset(args.paramFile, val_fnums,
            xptCacheDir=args.xptCacheDir)
    
    train_crop = XPointPatchDataset(train_dataset, patch=64, pos_ratio=0.6, retries=20)
    val_crop   = XPointPatchDataset(val_dataset, patch=64, pos_ratio=0.6, retries=20)

    t1 = timer()
    print("time (s) to create gkyl data loader: " + str(t1-t0))
    print(f"number of training frames (original + augmented): {len(train_dataset)}")
    print(f"number of validation frames: {len(val_dataset)}")

    train_loader = DataLoader(train_crop, batch_size=args.batchSize, shuffle=False)
    val_loader   = DataLoader(val_crop,   batch_size=args.batchSize, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(input_channels=4, base_channels=64).to(device)

    criterion = DiceLoss(smooth=1.0)
    
    # Reduce learning rate for mixed precision training
    effective_lr = args.learningRate
    if args.use_amp:
        # Less aggressive reduction for AMP
        effective_lr = args.learningRate * 0.5  # Reduce by 2x instead of 10x
        print(f"Adjusting learning rate for AMP: {args.learningRate} -> {effective_lr}")
    
    optimizer = optim.Adam(model.parameters(), lr=effective_lr, eps=1e-4)  # Higher epsilon for stability
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Initialize GradScaler for mixed precision if enabled
    scaler = None
    amp_dtype = torch.float16  # default
    
    if args.use_amp and torch.cuda.is_available():
        if args.amp_dtype == 'bfloat16':
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
                print("Using bfloat16 for mixed precision (no GradScaler needed)")
            else:
                print("Warning: bfloat16 not supported on this GPU, falling back to float16")
                amp_dtype = torch.float16
        
        # Only use GradScaler with float16
        if amp_dtype == torch.float16:
            # Initialize with very conservative settings for stability
            scaler = GradScaler(
                device='cuda',
                init_scale=2.**4,     # Much smaller initial scale (16 instead of 256)
                growth_factor=1.5,    # Slower growth
                backoff_factor=0.5,
                growth_interval=200,  # Wait longer before increasing scale
                enabled=True
            )
            print("Initialized GradScaler with very conservative settings for stability")
        
        print(f"Using Automatic Mixed Precision with {amp_dtype}")

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint_path = os.path.join(checkpoint_dir, "xpoint_model_latest.pt")
    start_epoch = 0
    train_loss = []
    val_loss = []

    if os.path.exists(latest_checkpoint_path):
        model, optimizer, start_epoch, train_loss, val_loss, scaler = load_model_checkpoint(
            model, optimizer, latest_checkpoint_path, scaler
        )
        print(f"Resuming training from epoch {start_epoch+1}")
    else:
        print("Starting training from scratch")

    t2 = timer()
    print("time (s) to prepare model: " + str(t2-t1))
    if args.use_amp:
        print(f"Using Automatic Mixed Precision (AMP) training with {amp_dtype}")

    num_epochs = args.epochs
    for epoch in range(start_epoch, num_epochs):
        train_loss.append(train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, args.use_amp, amp_dtype))
        val_loss.append(validate_one_epoch(model, val_loader, criterion, device, args.use_amp, amp_dtype))
        
        # Update learning rate based on validation loss
        if not np.isnan(val_loss[-1]):
            scheduler.step(val_loss[-1])
        
        print(f"[Epoch {epoch+1}/{num_epochs}]  TrainLoss={train_loss[-1]} ValLoss={val_loss[-1]}")
        
        # Save model checkpoint after each epoch
        if epoch % args.checkPointFrequency == 0:
            save_model_checkpoint(model, optimizer, train_loss, val_loss, epoch+1, checkpoint_dir, scaler)

    plot_training_history(train_loss, val_loss)
    print("time (s) to train model: " + str(timer()-t2))

    requiredLossDecreaseMagnitude = args.minTrainingLoss
    if len(train_loss) > 0 and train_loss[-1] > 0 and not np.isnan(train_loss[0]) and not np.isnan(train_loss[-1]):
        if np.log10(abs(train_loss[0]/train_loss[-1])) < requiredLossDecreaseMagnitude:
            print(f"TrainLoss reduced by less than {requiredLossDecreaseMagnitude} orders of magnitude: "
                  f"initial {train_loss[0]} final {train_loss[-1]} ... exiting")
            return 1
    else:
        print("Warning: Unable to check training loss reduction due to NaN or zero values")

    # (D) Plotting after training
    model.eval() # switch to inference mode
    outDir = "plots"
    interpFac = 1  

    # Evaluate on combined set for demonstration
    full_dataset = [train_dataset, val_dataset]

    t4 = timer()

    with torch.no_grad():
        for dataset in full_dataset:
            for item in dataset:
                fnum = item["fnum"]
                rotation = item["rotation"]
                reflectionAxis = item["reflectionAxis"]
                psi_np = item["psi"].numpy()[0]
                mask_gt = item["mask"].numpy()[0]
                x = item["x"]
                y = item["y"]
                filenameBase = item["filenameBase"]
                params = item["params"]

                # Get CNN prediction
                all_torch = item["all"].unsqueeze(0).to(device)
                
                if args.use_amp:
                    with autocast(device_type='cuda', dtype=amp_dtype):
                        pred_mask = model(all_torch)
                else:
                    pred_mask = model(all_torch)
                    
                pred_mask_np = pred_mask[0,0].cpu().numpy()
                # Binarize
                pred_bin = (pred_mask_np > 0.5).astype(np.float32)

                pred_prob = torch.sigmoid(pred_mask)
                pred_prob_np = (pred_prob > 0.5).float().cpu().numpy()

                pred_mask_bin = (pred_prob_np > 0.5).astype(np.float32)

                print(f"Frame {fnum} rotation {rotation} reflectionAxis {reflectionAxis}:")
                print(f"psi shape: {psi_np.shape}, min: {psi_np.min()}, max: {psi_np.max()}")
                print(f"pred_bin shape: {pred_bin.shape}, min: {pred_bin.min()}, max: {pred_bin.max()}")
                print(f"  Logits - min: {pred_mask_np.min():.5f}, max: {pred_mask_np.max():.5f}, mean: {pred_mask_np.mean():.5f}")
                print(f"  Probabilities (after sigmoid) - min: {pred_prob_np.min():.5f}, max: {pred_prob_np.max():.5f}, mean: {pred_prob_np.mean():.5f}")
                print(f"  Binary Mask (X-points) - count of 1s: {np.sum(pred_mask_bin)} / {pred_mask_bin.size} pixels")
                print(f"  Binary Mask (X_points) - shape: {pred_mask_bin.shape}, min: {pred_mask_bin.min()}, max: {pred_mask_bin.max()}")

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
                        xpoint_mask=np.squeeze(pred_mask_bin),
                        titleExtra="CNNXpoints",
                        outDir=outDir,
                        saveFig=True
                    )

                    pred_prob_np_full = pred_prob.cpu().numpy()
                    plot_model_performance(
                        psi_np, pred_prob_np_full, mask_gt, x, y, params, fnum, filenameBase,
                        outDir=outDir,
                        saveFig=True
                    )

    t5 = timer()
    print("time (s) to apply model: " + str(t5-t4))
    print("total time (s): " + str(t5-t0))

if __name__ == "__main__":
    main()