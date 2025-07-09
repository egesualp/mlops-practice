import torch
from lmu_ss25_mlops.model import MyAwesomeModel
from lmu_ss25_mlops.data import corrupt_mnist
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


def test_training_step_runs():
    model = MyAwesomeModel()
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    train_set, _ = corrupt_mnist()
    dataloader = DataLoader(train_set, batch_size=32, shuffle=True)

    model.train()
    batch = next(iter(dataloader))
    x, y = batch

    # Forward
    preds = model(x)
    loss = loss_fn(preds, y)

    # Backward
    loss.backward()
    optimizer.step()

    # Just assert loss is a valid scalar
    assert loss.item() > 0
