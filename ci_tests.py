import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import sys
# Local import within the function itself, which is a bit clunky
# but the original code did it this way.
# from XPointMLTest import validate_one_epoch 

class SyntheticXPointDataset(Dataset):
    """
    Synthetic dataset for CI testing that doesn't require actual simulation data.
    Creates predictable X-point patterns for testing model training pipeline.
    """
    def __init__(self, nframes=2, shape=(64, 64), nxpoints=4, seed=42):
        """
        nframes: Number of synthetic frames to generate
        shape: Spatial dimensions (H, W) of each frame
        nxpoints: Approximate number of X-points per frame
        seed: Random seed for reproducibility
        """
        super().__init__()
        self.nframes = nframes
        self.shape = shape
        self.nxpoints = nxpoints
        self.rng = np.random.RandomState(seed)
        
        #pre-generate all frames for consistency
        self.data = []
        for i in range(nframes):
            frame_data = self._generate_frame(i)
            self.data.append(frame_data)
    
    def _generate_frame(self, idx):
        """Generate a single synthetic frame with X-points"""
        H, W = self.shape
        
        #create synthetic psi field with some structure
        x = np.linspace(-np.pi, np.pi, W)
        y = np.linspace(-np.pi, np.pi, H)
        X, Y = np.meshgrid(x, y)
        
        #create a field with saddle points (X-points)
        psi = np.sin(X + 0.1*idx) * np.cos(Y + 0.1*idx) + \
              0.5 * np.sin(2*X) * np.cos(2*Y)
        
        # add some noise
        psi += 0.1 * self.rng.randn(H, W)
        
        #create synthetic B fields (derivatives of psi)
        bx = np.gradient(psi, axis=0)
        by = -np.gradient(psi, axis=1)
        
        #create synthetic current (Laplacian of psi)
        jz = -(np.gradient(np.gradient(psi, axis=0), axis=0) + 
                np.gradient(np.gradient(psi, axis=1), axis=1))
        
        # create X-point mask
        mask = np.zeros((H, W), dtype=np.float32)
        
        for _ in range(self.nxpoints):
            x_loc = self.rng.randint(10, W-10)
            y_loc = self.rng.randint(10, H-10)
            # Create 9x9 region around X-point
            mask[max(0, y_loc-4):min(H, y_loc+5), 
                 max(0, x_loc-4):min(W, x_loc+5)] = 1.0
        
        #Convert to torch tensors
        psi_torch = torch.from_numpy(psi.astype(np.float32)).unsqueeze(0)
        bx_torch = torch.from_numpy(bx.astype(np.float32)).unsqueeze(0)
        by_torch = torch.from_numpy(by.astype(np.float32)).unsqueeze(0)
        jz_torch = torch.from_numpy(jz.astype(np.float32)).unsqueeze(0)
        all_torch = torch.cat((psi_torch, bx_torch, by_torch, jz_torch))
        mask_torch = torch.from_numpy(mask).float().unsqueeze(0)
        
        x_coords = np.linspace(0, 1, W)
        y_coords = np.linspace(0, 1, H)
        
        params = {
            "axesNorm": 1.0, "plotContours": 1, "colorContours": 'k',
            "numContours": 50, "axisEqual": 1, "symBar": 1, "colormap": 'bwr'
        }
        
        return {
            "fnum": idx, "rotation": 0, "reflectionAxis": -1, "psi": psi_torch,
            "all": all_torch, "mask": mask_torch, "x": x_coords, "y": y_coords,
            "filenameBase": f"synthetic_frame_{idx}", "params": params
        }
    
    def __len__(self):
        return self.nframes
    
    def __getitem__(self, idx):
        return self.data[idx]

def test_checkpoint_functionality(model, optimizer, device, val_loader, criterion, scaler, UNet, Adam):
    """
    Test that model can be saved and loaded correctly.
    Returns True if test passes, False otherwise.
    
    """
    # Import locally to prevent circular dependency
    from XPointMLTest import validate_one_epoch, autocast
    
    print("\n" + "="*60)
    print("TESTING CHECKPOINT SAVE/LOAD FUNCTIONALITY")
    print("="*60)
    
    # Get the AMP settings from the model's current state to pass to validate_one_epoch
    use_amp = isinstance(scaler, torch.cuda.amp.GradScaler)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Get initial validation loss
    model.eval()
    initial_loss = validate_one_epoch(model, val_loader, criterion, device, use_amp, amp_dtype)
    print(f"Initial validation loss: {initial_loss:.6f}")
    
    # Save checkpoint with the correct AMP components
    test_checkpoint_path = "smoke_test_checkpoint.pt"
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': initial_loss,
        'test_value': 42
    }
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    torch.save(checkpoint, test_checkpoint_path)
    print(f"Saved checkpoint to {test_checkpoint_path}")
    
    # create new model and optimizer
    # NOTE: The base_channels here should match the original model's base_channels (32).
    # You had 64, which would cause an error later. Changed to 32.
    model2 = UNet(input_channels=4, base_channels=32).to(device) 
    optimizer2 = Adam(model2.parameters(), lr=1e-5)
    
    # load checkpoint
    loaded_checkpoint = torch.load(test_checkpoint_path, map_location=device, weights_only=False)
    model2.load_state_dict(loaded_checkpoint['model_state_dict'])
    optimizer2.load_state_dict(loaded_checkpoint['optimizer_state_dict'])

    # Load the scaler state if present
    scaler2 = None
    if 'scaler_state_dict' in loaded_checkpoint:
        scaler2 = torch.cuda.amp.GradScaler()
        scaler2.load_state_dict(loaded_checkpoint['scaler_state_dict'])
    
    assert loaded_checkpoint['test_value'] == 42, "Test value mismatch!"
    print("Checkpoint test value verified")
    
    #get loaded model validation loss
    model2.eval()
    # Now pass the AMP arguments to validate_one_epoch
    loaded_loss = validate_one_epoch(model2, val_loader, criterion, device, use_amp, amp_dtype)
    print(f"Loaded model validation loss: {loaded_loss:.6f}")
    
    # check if losses match
    loss_diff = abs(initial_loss - loaded_loss)
    success = loss_diff < 1e-6
    if success:
        print(f"✓ Checkpoint test PASSED (loss difference: {loss_diff:.2e})")
    else:
        print(f"✗ Checkpoint test FAILED (loss difference: {loss_diff:.2e})")

    if os.path.exists(test_checkpoint_path):
        os.remove(test_checkpoint_path)
        print(f"Cleaned up {test_checkpoint_path}")
    
    print("="*60 + "\n")
    return success