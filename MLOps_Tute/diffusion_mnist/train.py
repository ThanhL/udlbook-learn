import os

import hydra
import torch
import logging
import torch.utils.data as data
from torchvision import transforms, datasets

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from omegaconf.omegaconf import OmegaConf
from model import UNetDiffSystem


# -- Main
@hydra.main(version_base=None,
            config_path="./configs",
            config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)

    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

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
    train_set_size = int(len(train_set) * cfg.training.train_val_split)
    valid_set_size = len(train_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(cfg.training.seed)
    train_set, valid_set = data.random_split(train_set,
                                             [train_set_size, valid_set_size],
                                             generator=seed)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.train_dataloader.batch_size,
        shuffle=cfg.train_dataloader.shuffle,
        num_workers=cfg.train_dataloader.num_workers,
        # pin_memory=cfg.train_dataloader.pin_memory,
        # persistent_workers=cfg.train_dataloader.persistent_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=cfg.val_dataloader.batch_size,
        num_workers=cfg.val_dataloader.num_workers,
        shuffle=cfg.val_dataloader.shuffle,
        # pin_memory=cfg.val_dataloader.pin_memory,
        # persistent_workers=cfg.val_dataloader.persistent_workers,
    )

    # -- Model setup
    # Model
    model = UNetDiffSystem(
        num_diffusion_iters=cfg.diffusion.noise_scheduler.num_train_timesteps,
        learning_rate=cfg.optimizer.lr
    )

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
        max_epochs=cfg.training.max_epochs,
        fast_dev_run=False,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
        ],
    )
    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader)


if __name__ == "__main__":
    main()
