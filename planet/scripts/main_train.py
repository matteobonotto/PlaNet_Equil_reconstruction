from typing import List, Optional
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import WandbLogger


from planet.train import LightningPlaNet, DataModule
from planet.utils import (
    parse_arguments, load_config, get_accelerator, last_ckp_path, save_model_and_scaler
)


if __name__ == "__main__":

    args = parse_arguments()
    config = load_config(args.config)

    #
    save_dir = Path(config.save_path).parent
    save_dir.mkdir(exist_ok=True, parents=True)

    ### instantiate model and datamodule
    model = LightningPlaNet(config=config)
    datamodule = DataModule(config=config)

    ### define some callbacks
    callbacks: List[Callback] = []
    if config.save_checkpoints is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=save_dir / Path("ckp"), save_top_k=2, monitor="val_loss"
            )
        )

    # get the logger
    if config.log_to_wandb:
        wandb_logger = WandbLogger(project=config.wandb_project)

    ### train the model
    trainer = Trainer(
        max_epochs=config.epochs,
        accelerator=get_accelerator(),
        devices="auto",
        callbacks=callbacks,
        logger=wandb_logger,
    )
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=(
            last_ckp_path(save_dir / Path("ckp"))
            if config.resume_from_checkpoint
            else None
        ),
    )

    ### save model + scaler for inference
    save_model_and_scaler(trainer, datamodule, config)
