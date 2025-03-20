import numpy as np
import matplotlib.pyplot as plt
import os

from utils import gkData
from utils import auxFuncs
from utils import plotParams

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
            self.data.append(self.load(fnum))

    def __len__(self):
        return len(self.fnumList)

    def __getitem__(self, idx):
        fnum = self.fnumList[idx]
        return self.data[idx]

    def load(self, fnum):
        t0 = timer()

        # Initialize gkData object
        useB        = 0
        varid       = "psi"
        tmp = gkData.gkData(self.paramFile, fnum, varid, self.params)
        print("time (s) to read gkyl data from disk: " + str(timer()-t0))

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

        t2 = timer()
        # Indicies of critical points, X points, and O points (max and min)
        critPoints = auxFuncs.getCritPoints(f, g=g, dx=dx)
        [xpts, optsMax, optsMin] = auxFuncs.getXOPoints(f, critPoints, g=g, dx=dx)
        print("time (s) to find X and O points: " + str(timer()-t2))

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

        print("time (s) to load and process gkyl frame: " + str(timer()-t0))

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
      in:  (N, 1,   H, W)   ++++ BX, BY, JZ
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

        # Flatten both inputs and targets
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
            x / params["axesNorm"][0],
            y / params["axesNorm"][1],
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
            x[tp_rows] / params["axesNorm"][0],
            y[tp_cols] / params["axesNorm"][1],
            'o', color='green', markersize=8, label="True Positives"
        )
    
    if len(fp_rows) > 0:
        plt.plot(
            x[fp_rows] / params["axesNorm"][0],
            y[fp_cols] / params["axesNorm"][1],
            'o', color='red', markersize=8, label="False Positives"
        )
    
    if len(fn_rows) > 0:
        plt.plot(
            x[fn_rows] / params["axesNorm"][0],
            y[fn_cols] / params["axesNorm"][1],
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

def plot_training_history(train_losses, val_losses, save_path='output_images/training_history.png'):
    """
    Plots training and validation losses across epochs.
    
    Parameters:
    train_losses (list): List of training losses for each epoch
    val_losses (list): List of validation losses for each epoch
    save_path (str): Path to save the resulting plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add some padding to y-axis to make visualization clearer
    ymin = min(min(train_losses), min(val_losses)) * 0.9
    ymax = max(max(train_losses), max(val_losses)) * 1.1
    plt.ylim(ymin, ymax)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Training history plot saved to {save_path}")
    plt.close()

def main():
    t0 = timer()
    paramFile = '/space/cwsmith/nsfCssiSpaceWeather2022/mlReconnection2025/1024Res_v0/pkpm_2d_turb_p2-params.txt'

    train_fnums = range(1, 140)
    val_fnums   = range(141, 150)

    train_dataset = XPointDataset(paramFile, train_fnums, constructJz=1, interpFac=1, saveFig=1)
    val_dataset   = XPointDataset(paramFile, val_fnums,   constructJz=1, interpFac=1, saveFig=1)

    t1 = timer()
    print("time (s) to create gkyl data loader: " + str(t1-t0))

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(input_channels=1, base_channels=16).to(device)
    criterion = FocalLoss(alpha=0.999, gamma=10)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    t2 = timer()
    print("time (s) to prepare model: " + str(t2-t1))

    train_loss = []
    val_loss = []
    
    num_epochs = 200
    for epoch in range(num_epochs):
        train_loss.append(train_one_epoch(model, train_loader, criterion, optimizer, device))
        val_loss.append(validate_one_epoch(model, val_loader, criterion, device))
        print(f"[Epoch {epoch+1}/{num_epochs}]  TrainLoss={train_loss[-1]} ValLoss={val_loss[-1]}")

    print("time (s) to train model: " + str(timer()-t2))

    requiredLossDecreaseMagnitude = 3;
    if np.log10(abs(train_loss[0]/train_loss[-1])) < requiredLossDecreaseMagnitude:
        print(f"TrainLoss reduced by less than {requiredLossDecreaseMagnitude} orders of magnitude: "
              f"initial {train_loss[0]} final {train_loss[-1]} ... exiting")
        return 1;

    # (D) Plotting after training
    model.eval()    # Find out what this means
    outDir = "output_images"
    os.makedirs(outDir, exist_ok=True)
    interpFac = 1  

    # Evaluate on combined set for demonstration. Exam this part to see if save to remove
    full_fnums = list(train_fnums) + list(val_fnums)
    full_dataset = [train_dataset, val_dataset]

    t4 = timer()

    with torch.no_grad():
      for set in full_dataset:
        for item in set:
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
