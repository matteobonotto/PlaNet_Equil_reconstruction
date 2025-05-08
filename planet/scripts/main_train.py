from typing import List, Optional
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
import re
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import WandbLogger

from argparse import ArgumentParser, Namespace
import yaml

from planet.train import LightningPlaNet, DataModule
from planet.config import PlaNetConfig


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("config", help="path to config file")
    args, _ = parser.parse_known_args()
    return args


def load_config(path: str) -> PlaNetConfig:
    config_dict = yaml.safe_load(open(path, "r"))
    return PlaNetConfig.from_dict(config_dict=config_dict)


def last_ckp_path(ckpt_path: str) -> Path:
    # for ckp in Path(ckpt_path).iterdir():
    # Regex to extract epoch and step
    pattern = re.compile(r"epoch=(\d+)-step=(\d+)")

    # Extract (epoch, step) tuples + path
    parsed = []
    for path in Path(ckpt_path).iterdir():
        match = pattern.search(path.name)
        if match:
            epoch, step = map(int, match.groups())
            parsed.append(((epoch, step), path))

    # Find the path with max (epoch, step)
    _, latest = max(parsed, key=lambda x: (x[0][0], x[0][1]))
    return latest

def save_model_and_scaler(trainer: Trainer, datamodule: DataModule, config: PlaNetConfig) -> None:
    model = trainer.model.model
    model.eval()
    torch.save(trainer.model.state_dict(), config.save_path)
    scaler = datamodule.dataset.scaler
    with open(save_dir / Path('scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

def get_accelerator() -> Optional[str]:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return None

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
        callbacks.append(ModelCheckpoint(
            dirpath=save_dir / Path('ckp'), 
            save_top_k=2, 
            monitor="val_loss"
        ))

    # get the logger
    if config.log_to_wandb:
        wandb_logger = WandbLogger(project=config.wandb_project)

    ### train the model
    trainer = Trainer(
        max_epochs=config.epochs,
        accelerator=get_accelerator(),
        devices='auto', 
        callbacks=callbacks,
        logger=wandb_logger,
    )
    trainer.fit(
        model=model, 
        datamodule=datamodule,
        ckpt_path=last_ckp_path(save_dir / Path('ckp')) if config.resume_from_checkpoint else None
    )

    ### save model + scaler for inference
    save_model_and_scaler(trainer, datamodule, config)

    