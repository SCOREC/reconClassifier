import numpy as np
import matplotlib.pyplot as plt
import os

from utils import gkData
from utils import auxFuncs
from utils import plotParams

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from timeit import default_timer as timer

# DATASET DEFINITION
class XPointDataset(Dataset):
    """
    Dataset that processes frames in [fnumList]. For each frame (fnum):
      - Sets up "params" according to your snippet.
      - Reads psi from gkData (varid='psi'), possibly jz if needed.
      - Interpolates if interpFac>1.
      - Finds X-points -> builds a 2D binary mask.
      - Returns (psiTensor, maskTensor) as a PyTorch (float) pair.
    """
    def __init__(self, paramFile, fnumList, constructJz=1, interpFac=1, saveFig=1):
        """
        paramFile:   Path to parameter file (string).
        fnumList:    List of frames to iterate. 
        constructJz: Whether to compute jz from second derivatives or load from gkData.
        interpFac:   Interpolation factor for FFT-based upsampling.
        saveFig:     If True, we might save intermediate plots (optional).
        """
        super().__init__()
        self.paramFile   = paramFile
        self.fnumList    = list(fnumList)  # ensure indexable
        self.constructJz = constructJz
        self.interpFac   = interpFac
        self.saveFig     = saveFig

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

        # output directory:
        self.outDir = "plots"
        os.makedirs(self.outDir, exist_ok=True)

        # load all the data
        self.data = []
        for fnum in fnumList:
            print("fnum " + str(fnum))
            self.data.append(self.load(fnum))

    def __len__(self):
        return len(self.fnumList)

    def __getitem__(self, idx):
        t0 = timer()
        fnum = self.fnumList[idx]
        print(f"[XPointDataset] Fetching fileNum = {fnum}")
        return self.data[idx]

    def load(self, fnum):
        t0 = timer()
        print(f"[XPointDataset] Processing fileNum = {fnum}")

        # Initialize gkData object
        interpFac   = 1
        useB        = 0
        varid       = "psi"
        tmp = gkData.gkData(self.paramFile, fnum, varid, self.params)

        refSpeciesAxes  = 'ion'
        refSpeciesAxes2 = 'ion'
        refSpeciesTime  = 'ion'

        speciesIndexAxes  = tmp.speciesFileIndex.index(refSpeciesAxes)
        speciesIndexAxes2 = tmp.speciesFileIndex.index(refSpeciesAxes2)
        speciesIndexTime  = tmp.speciesFileIndex.index(refSpeciesTime)

        # Overwrite these normalizations in params per-frame
        self.params["axesNorm"] = [
            tmp.d[speciesIndexAxes],
            tmp.d[speciesIndexAxes],
            tmp.vt[speciesIndexAxes2],
            tmp.vt[speciesIndexAxes2],
            tmp.vt[speciesIndexAxes2]
        ]
        self.params["timeNorm"] = tmp.omegaC[speciesIndexTime]
        self.params["axesLabels"] = ['$x/d_i$', '$y/d_i$', '$z/d_p$']

        varPsi = gkData.gkData(self.paramFile, fnum, 'psi', self.params).compactRead()
        psi_raw = varPsi.data
        coords0 = varPsi.coords

        print(f"   psi shape: {psi_raw.shape}, min={psi_raw.min()}, max={psi_raw.max()}")

        # -------------- 4) Interpolate if interpFac>1 --------------
        if self.interpFac > 1:
            psi, coords = auxFuncs.getFFTInterp(psi_raw, coords0, fac=self.interpFac)
        else:
            print("   Using original (no upsampling)")
            psi = psi_raw
            coords = coords0

        x = coords[0]
        y = coords[1]
        dx = [coords[d][1] - coords[d][0] for d in range(2)]

        # -------------- 5) Find X-points --------------
        # Suppose you have an auxFuncs.findXPoints(...) 
        # that returns an array of shape (nXpts, 2), each row = (ix, iy)

        if useB:
            f = bx;
            g = by
        else:
            f = psi;
            g = None

        # Indicies of critical points, X points, and O points (max and min)
        critPoints = auxFuncs.getCritPoints(f, g=g, dx=dx)
        [xpts, optsMax, optsMin] = auxFuncs.getXOPoints(f, critPoints, g=g, dx=dx)

        numC = np.shape(critPoints)[1]
        numX = np.shape(xpts)[0];
        numOMax = np.shape(optsMax)[0];
        numOMin = np.shape(optsMin)[0];

        # Create array of 0s with 1s only at X points
        binaryMap = np.zeros(np.shape(f));
        binaryMap[xpts[:, 0], xpts[:, 1]] = 1

        # -------------- 6) Convert to Torch Tensors --------------
        psi_torch = torch.from_numpy(psi).float().unsqueeze(0)      # [1, Nx, Ny]
        mask_torch = torch.from_numpy(binaryMap).float().unsqueeze(0)  # [1, Nx, Ny]

        t1 = timer()
        print("time (s) to get gkyl data: " + str(t1-t0))

        return {
            "fnum": fnum,
            "psi": psi_torch,        # shape [1, Nx, Ny]
            "mask": mask_torch,      # shape [1, Nx, Ny]    // Maybe redundant, 
            "psi_np": psi,           # 2D np array [Nx, Ny]
            "mask_np": binaryMap,    # 2D np array [Nx, Ny]
            "x": x,
            "y": y,
            "filenameBase": tmp.filenameBase, 
            "params": dict(self.params)  # copy of the params for local plotting
        }



# 2) U-NET ARCHITECTURE
class UNet(nn.Module):
    """
    A simplified U-Net for binary segmentation:
      in:  (N, 1,   H, W)
      out: (N, 1,   H, W)
    """
    def __init__(self, input_channels=1, base_channels=16):
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

    def double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)               # shape: [N, base_channels, H, W]
        p1 = self.pool(e1)             # half spatial dims

        e2 = self.enc2(p1)             # [N, 2*base_channels, H/2, W/2]
        p2 = self.pool(e2)

        e3 = self.enc3(p2)             # [N, 4*base_channels, H/4, W/4]
        p3 = self.pool(e3)             # [N, 4*base_channels, H/8, W/8]

        # Bottleneck
        b  = self.bottleneck(p3)       # [N, 8*base_channels, H/8, W/8]

        # Decoder
        u3 = self.up3(b)               # -> shape ~ e3
        cat3 = torch.cat([u3, e3], dim=1)  # skip connection
        d3 = self.dec3(cat3)

        u2 = self.up2(d3)              # -> shape ~ e2
        cat2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(cat2)

        u1 = self.up1(d2)              # -> shape ~ e1
        cat1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(cat1)

        out = self.out_conv(d1)
        return out  # We'll apply sigmoid in the loss or after
    

# TRAIN & VALIDATION UTILS
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in loader:
        psi, mask = batch["psi"].to(device), batch["mask"].to(device)
        pred = model(psi)
        loss = criterion(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            psi, mask = batch["psi"].to(device), batch["mask"].to(device)
            pred = model(psi)
            loss = criterion(pred, mask)
            val_loss += loss.item()
    return val_loss / len(loader)

# PLOTTING FUNCTION
def plot_psi_contours_and_xpoints(psi_np, x, y, params, fnum, filenameBase, interpFac,
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
            x / params["axesNorm"][0],
            y / params["axesNorm"][1],
            np.transpose(psi_np),          
            params["numContours"],
            colors=params["colorContours"],
            linewidths=0.75
        )

    plt.xlabel(r"$x/d_i$")
    plt.ylabel(r"$y/d_i$")
    if params["axisEqual"]:
        plt.gca().set_aspect("equal", "box")

    plt.title(f"Vector Potential Contours {titleExtra}, fileNum={fnum}")

    # Overlay X-points if xpoint_mask is given
    if xpoint_mask is not None:
        # find where xpoint_mask == 1
        xpts_row, xpts_col = np.where(xpoint_mask == 1)
        # plot as black 'x'
        plt.plot(
            x[xpts_row] / params["axesNorm"][0],
            y[xpts_col] / params["axesNorm"][1],
            'xk'
        )

    # Save the figure if needed (could be removed as we save anyway)
    if saveFig:
        basename = os.path.basename(filenameBase)
        saveFilename = os.path.join(
            outDir,
            f"{basename}_interpFac_{interpFac}_{fnum:04d}{titleExtra.replace(' ','_')}.png"
        )
        plt.savefig(saveFilename, dpi=300)
        print("   Figure written to", saveFilename)

    plt.close()


def main():
    t0 = timer()
    paramFile = '/space/cwsmith/nsfCssiSpaceWeather2022/mlReconnection2025/1024Res_v0/pkpm_2d_turb_p2-params.txt'

    train_fnums = range(75, 101)  
    val_fnums   = range(101, 106)  

    train_dataset = XPointDataset(paramFile, train_fnums, constructJz=1, interpFac=1, saveFig=1)
    val_dataset   = XPointDataset(paramFile, val_fnums,   constructJz=1, interpFac=1, saveFig=1)

    t1 = timer()
    print("time (s) to create gkyl data loader: " + str(t1-t0))

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(input_channels=1, base_channels=16).to(device)
    pos_weight = torch.tensor([2000.0], dtype=torch.float, device=device)   # Pos_weight: a tuning parameter, to add weight to X point so it becomes more evident
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    t2 = timer()
    print("time (s) to prepare model: " + str(t2-t1))
    
    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = validate_one_epoch(model, val_loader, criterion, device)
        print(f"[Epoch {epoch+1}/{num_epochs}]  TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}")

    t3 = timer()
    print("time (s) to train model: " + str(t3-t2))

    # (D) Plotting after training
    model.eval()    # Find out what this means
    outDir = "output_images"
    os.makedirs(outDir, exist_ok=True)
    interpFac = 1  

    # Evaluate on combined set for demonstration. Exam this part to see if save to remove
    full_fnums = list(train_fnums) + list(val_fnums)
    full_dataset = XPointDataset(paramFile, full_fnums, constructJz=1, interpFac=interpFac, saveFig=1)  

    t4 = timer()
    print("time (s) to create gkyl data loader: " + str(t4-t3))

    with torch.no_grad():
        for item in full_dataset:
            # item is a dict with keys: fnum, psi, mask, psi_np, mask_np, x, y, tmp, params
            fnum     = item["fnum"]
            psi_np   = item["psi_np"]
            mask_gt  = item["mask_np"]
            x        = item["x"]
            y        = item["y"]
            filenameBase      = item["filenameBase"]
            params   = item["params"]

            # Get CNN prediction
            psi_torch = item["psi"].unsqueeze(0).to(device) # => [1,1,Nx,Ny]
            pred_mask = model(psi_torch)                    # => [1,1,Nx,Ny]
            pred_mask_np = pred_mask[0,0].cpu().numpy()     # => [Nx,Ny]
            # Binarize
            pred_bin = (pred_mask_np > 0.5).astype(np.float32)

            pred_prob = torch.sigmoid(pred_mask)
            pred_prob_np = (pred_prob > 0.5).float().cpu().numpy()

            pred_mask_bin = (pred_prob_np > 0.5).astype(np.float32)  # Thresholding at 0.5, can be fine tune

            print(f"Frame {fnum}:")
            print(f"psi shape: {psi_np.shape}, min: {psi_np.min()}, max: {psi_np.max()}")
            print(f"pred_bin shape: {pred_bin.shape}, min: {pred_bin.min()}, max: {pred_bin.max()}")
            print(f"  Logits - min: {pred_mask_np.min():.5f}, max: {pred_mask_np.max():.5f}, mean: {pred_mask_np.mean():.5f}")
            print(f"  Probabilities (after sigmoid) - min: {pred_prob_np.min():.5f}, max: {pred_prob_np.max():.5f}, mean: {pred_prob_np.mean():.5f}")
            print(f"  Binary Mask (X-points) - count of 1s: {np.sum(pred_mask_bin)} / {pred_mask_bin.size} pixels")
            print(f"  Binary Mask (X_points) - shape: {pred_mask_bin.shape}, min: {pred_mask_bin.min()}, max: {pred_mask_bin.max()}")

            # Plot GROUND TRUTH
            plot_psi_contours_and_xpoints(
                psi_np, x, y, params, fnum, filenameBase, interpFac,
                xpoint_mask=mask_gt,
                titleExtra="(GT X-points)",
                outDir=outDir,
                saveFig=True
            )

            # Plot CNN PREDICTIONS
            plot_psi_contours_and_xpoints(
                psi_np, x, y, params, fnum, filenameBase, interpFac,
                xpoint_mask=np.squeeze(pred_mask_bin),
                titleExtra="(CNN X-points)",
                outDir=outDir,
                saveFig=True
            )

    t5 = timer()
    print("time (s) to apply model: " + str(t5-t4))

if __name__ == "__main__":
    main()
