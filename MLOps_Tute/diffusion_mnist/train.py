import os
from scipy import datasets
import torch
import torch.utils.data as data
from torchvision import transforms, datasets

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from model import UNetDiffSystem


def main():
    # -- Data setup
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root=os.getcwd(),
                               download=True,
                               train=True,
                               transform=transform)
    test_set = datasets.MNIST(root=os.getcwd(),
                              download=True,
                              train=False,
                              transform=transform)

    # use 20% of training data for validation
    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(train_set,
                                             [train_set_size, valid_set_size],
                                             generator=seed)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=256,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=256,
                                               shuffle=False)

    # -- Model setup
    # Model
    model = UNetDiffSystem(num_diffusion_iters=100, learning_rate=1e-4)

    # Checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min",
        # every_n_epochs=10,
        # save_last=True,
        save_on_train_epoch_end=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    # Loggers
    # logger = None
    logger = WandbLogger(project="MNIST_Diffusion")
    # logger = TensorBoardLogger("logs/", name="mnist", version=1)

    # Train
    trainer = L.Trainer(
        default_root_dir="logs",
        devices=(1 if torch.cuda.is_available() else 0),
        max_epochs=10,
        fast_dev_run=False,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            # early_stopping_callback,
        ],
    )
    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader)


if __name__ == "__main__":
    main()
