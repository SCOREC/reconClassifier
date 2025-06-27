import numpy as np
import matplotlib.pyplot as plt
import os, errno, sys, argparse
from pathlib import Path
from timeit import default_timer as timer
import sys
import argparse

from utils import gkData
from utils import auxFuncs
from utils import plotParams

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2  # rotate tensor

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
    mask = v2.functional.rotate(frameData["mask"], deg, v2.InterpolationMode.BILINEAR)
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
    psi = torch.flip(frameData["psi"][0], dims=(axis,)).unsqueeze(0)
    all = torch.flip(frameData["all"], dims=(axis,))
    mask = torch.flip(frameData["mask"][0], dims=(axis,)).unsqueeze(0)
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
        if self.xptCacheDir != None:
          if not self.xptCacheDir.is_dir():
              print(f"Xpoint cache directory {self.xptCacheDir} does not exist...  exiting")
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

        # -------------- 6) Convert to Torch Tensors --------------
        psi_torch = torch.from_numpy(fields["psi"]).float().unsqueeze(0)      # [1, Nx, Ny]
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
            "psi": psi_torch,        # shape [1, Nx, Ny]
            "all": all_torch,        # shape [4, Nx, Ny]
            "mask": mask_torch,      # shape [1, Nx, Ny]
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

    def __len__(self):
        # give each full frame K random crops per epoch (K=16 by default)
        return len(self.base_ds) * 16

    def _crop(self, arr, top, left):
        return arr[..., top:top+self.patch, left:left+self.patch]

    def __getitem__(self, _):
        frame = self.base_ds[self.rng.integers(len(self.base_ds))]
        H, W  = frame["mask"].shape[-2:]

        for attempt in range(self.retries):
            y0 = self.rng.integers(0, H - self.patch + 1)
            x0 = self.rng.integers(0, W - self.patch + 1)
            crop_mask = self._crop(frame["mask"], y0, x0)
            has_pos   = crop_mask.sum() > 0
            want_pos  = (attempt / self.retries) < self.pos_ratio

            if has_pos == want_pos or attempt == self.retries - 1:
                return {
                    "all" : self._crop(frame["all"],  y0, x0),
                    "mask": crop_mask
                }


class UNet(nn.Module):
    def __init__(self, input_channels=4, base=64):
        super().__init__()
        self.enc1 = self._dbl(input_channels, base, dilation=1)
        self.enc2 = self._dbl(base, base*2, dilation=1)
        self.enc3 = self._dbl(base*2, base*4, dilation=2)    # ← dilated
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = self._dbl(base*4, base*8, dilation=4)  # ← dilated
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = self._dbl(base*8, base*4, dilation=1)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = self._dbl(base*4, base*2, dilation=1)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = self._dbl(base*2, base, dilation=1)
        self.out = nn.Conv2d(base, 1, 1)

    @staticmethod
    def _dbl(inp, out, dilation=1):
        pad = dilation
        return nn.Sequential(
            nn.Conv2d(inp, out, 3, padding=pad, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(out, out, 3, padding=pad, dilation=dilation),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)



class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1); targets = targets.view(-1)
        inter = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = (2*inter + self.smooth) / (union + self.smooth)
        return 1 - dice


def make_criterion(pos_weight):
    bce  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).float())
    dice = DiceLoss()
    def _loss(logits, target):
        return 0.5 * bce(logits, target) + 0.5 * dice(logits, target)
    return _loss



@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); loss = 0
    for batch in loader:
        x, y = batch["all"].to(device), batch["mask"].to(device)
        logits = model(x)
        loss += criterion(logits, y).item()
    return loss / len(loader)


def train_epoch(model, loader, criterion, opt, device):
    model.train(); loss = 0
    for batch in loader:
        x, y = batch["all"].to(device), batch["mask"].to(device)
        opt.zero_grad()
        l = criterion(model(x), y)
        l.backward(); opt.step(); loss += l.item()
    return loss / len(loader)




def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paramFile", type=Path, required=True)
    p.add_argument("--xptCacheDir", type=Path)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    args = p.parse_args()

    # frame splits hard‑coded for demo
    train_fnums = range(1, 141)
    val_fnums   = range(141, 150)

    train_full = XPointDataset(args.paramFile, train_fnums,
                               xptCacheDir=args.xptCacheDir,
                               rotateAndReflect=True)
    val_full   = XPointDataset(args.paramFile, val_fnums,
                               xptCacheDir=args.xptCacheDir)

    train_ds = XPointPatchDataset(train_full, patch=64, pos_ratio=0.8) # 64 x 64 cropping
    val_ds   = XPointPatchDataset(val_full,   patch=64, pos_ratio=0.5)

    loader_tr = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    loader_va = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet().to(device)

    # estimate pos/neg for weight (rough):
    pos_px = 30 * 9 * 9
    neg_px = 1024*1024 - pos_px
    criterion = make_criterion(pos_weight=neg_px/pos_px)

    opt = optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs+1):
        tr_loss = train_epoch(model, loader_tr, criterion, opt, device)
        va_loss = evaluate(model, loader_va, criterion, device)
        print(f"Epoch {ep:03d}: train {tr_loss:.4f} | val {va_loss:.4f}")

        # quick checkpoint
        if ep % 10 == 0:
            torch.save(model.state_dict(), f"chkpt_ep{ep}.pt")

if __name__ == "__main__":
    main()
