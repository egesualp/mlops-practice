import torch
from lmu_ss25_mlops.data import preprocess_data
import os

def test_preprocess_data(tmp_path):
    # Simulate raw_dir
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    # Create dummy image/target tensors
    for i in range(6):
        torch.save(torch.randn(5000, 28, 28), raw_dir / f"train_images_{i}.pt")
        torch.save(torch.randint(0, 10, (5000,)), raw_dir / f"train_target_{i}.pt")

    torch.save(torch.randn(5000, 28, 28), raw_dir / "test_images.pt")
    torch.save(torch.randint(0, 10, (5000,)), raw_dir / "test_target.pt")

    # Run preprocessing
    preprocess_data(str(raw_dir), str(processed_dir))

    # Check that processed files exist and have correct shape
    train_images = torch.load(processed_dir / "train_images.pt")
    assert train_images.shape == (30000, 1, 28, 28), "Processed train images shape is incorrect"
    test_target = torch.load(processed_dir / "test_target.pt")
    assert test_target.shape == (5000,), "Processed test targets shape is incorrect"
