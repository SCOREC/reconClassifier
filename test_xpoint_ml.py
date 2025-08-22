import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import pytest

# Make sure all required functions are imported from the main file
from XPointMLTest import UNet, DiceLoss, expand_xpoints_mask, validate_one_epoch
from ci_tests import SyntheticXPointDataset

# --- Pytest Fixtures ---
@pytest.fixture
def unet_model():
    return UNet(input_channels=4, base_channels=16)

@pytest.fixture
def dice_loss():
    return DiceLoss()

@pytest.fixture
def synthetic_dataset():
    return SyntheticXPointDataset(nframes=2, shape=(32, 32))

@pytest.fixture
def synthetic_batch(synthetic_dataset):
    return synthetic_dataset[0]

# --- 1. Unit Tests (Utils & Loss Functions) ---
def test_expand_xpoints_mask():
    mask = np.zeros((20, 20))
    mask[10, 10] = 1
    expanded = expand_xpoints_mask(mask, kernel_size=5)
    assert expanded.shape == (20, 20)
    assert np.sum(expanded) == 25
    assert expanded[10, 10] == 1
    assert expanded[8, 8] == 1
    assert expanded[7, 7] == 0

def test_dice_loss_perfect_match(dice_loss):
    target = torch.ones(1, 1, 10, 10)
    logits = torch.full((1, 1, 10, 10), 10.0)  #large positive logits
    loss = dice_loss(logits, target)
    #due to smoothing factor, perfect match doesn't give exactly 0
    assert loss < 1e-4, f"Loss should be near 0, got {loss.item()}"

def test_dice_loss_no_match(dice_loss):
    target = torch.zeros(1, 1, 10, 10)
    logits = torch.full((1, 1, 10, 10), 10.0)
    loss = dice_loss(logits, target)
    expected_loss = 1.0 - (1.0 / (100 + 1.0))
    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-3)

# --- 2. Dataset Integrity Test ---
def test_synthetic_dataset_integrity(synthetic_dataset):
    assert len(synthetic_dataset) == 2
    item = synthetic_dataset[0]
    expected_keys = ["fnum", "all", "mask", "psi", "x", "y", "rotation", "reflectionAxis", "filenameBase", "params"]
    assert all(key in item for key in expected_keys)
    assert item['all'].shape == (4, 32, 32)
    assert item['mask'].shape == (1, 32, 32)
    assert item['psi'].shape == (1, 32, 32)
    assert item['all'].dtype == torch.float32
    assert item['mask'].dtype == torch.float32

# --- 3. Model Forward/Backward Pass Test ---
def test_model_forward_backward_pass(unet_model, synthetic_batch, dice_loss):
    model = unet_model
    loss_fn = dice_loss
    input_tensor = synthetic_batch['all'].unsqueeze(0)
    target_tensor = synthetic_batch['mask'].unsqueeze(0)
    prediction = model(input_tensor)
    assert prediction.shape == target_tensor.shape
    loss = loss_fn(prediction, target_tensor)
    assert loss.item() > 0
    loss.backward()
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads, "No gradients were computed during the backward pass."
    grad_sum = sum(p.grad.sum() for p in model.parameters() if p.grad is not None)
    assert grad_sum != 0, "Gradients are all zero."

# --- 4. Standalone checkpoint test for pytest ---
def test_checkpoint_save_load(unet_model, synthetic_dataset):
    """
    Standalone pytest version of checkpoint functionality test
    """
    device = torch.device("cpu")
    model = unet_model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = DiceLoss()
    
    # Create a simple dataloader
    val_loader = DataLoader(synthetic_dataset, batch_size=1, shuffle=False)
    
    # get initial loss, passing the required AMP arguments
    # we can assume no AMP for this CPU-based unit test
    initial_loss = validate_one_epoch(model, val_loader, criterion, device, use_amp=False, amp_dtype=torch.float32)
    
    # Save checkpoint
    test_checkpoint_path = "test_checkpoint_pytest.pt"
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': initial_loss,
        'test_value': 42
    }
    torch.save(checkpoint, test_checkpoint_path)
    
    # Create new model and load
    model2 = UNet(input_channels=4, base_channels=16).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-5)
    
    loaded_checkpoint = torch.load(test_checkpoint_path, map_location=device, weights_only=False)
    model2.load_state_dict(loaded_checkpoint['model_state_dict'])
    optimizer2.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    
    assert loaded_checkpoint['test_value'] == 42
    
    # Get loaded model loss, again passing the AMP arguments
    loaded_loss = validate_one_epoch(model2, val_loader, criterion, device, use_amp=False, amp_dtype=torch.float32)
    
    # Check if losses match
    loss_diff = abs(initial_loss - loaded_loss)
    assert loss_diff < 1e-6, f"Loss difference too large: {loss_diff}"
    
    # Cleanup
    if os.path.exists(test_checkpoint_path):
        os.remove(test_checkpoint_path)
        
def test_model_inference(unet_model, synthetic_batch):
    model = unet_model
    input_tensor = synthetic_batch['all'].unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    assert output.shape == (1, 1, 32, 32)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()