from argparse import ArgumentParser, Namespace
from typing import Optional
import yaml
from pathlib import Path
import re
import torch
import pickle
from lightning import Trainer

from .config import PlaNetConfig
from .train import DataModule


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


def save_model_and_scaler(
    trainer: Trainer, datamodule: DataModule, config: PlaNetConfig
) -> None:
    save_dir = Path(config.save_path).parent
    save_dir.mkdir(exist_ok=True, parents=True)
    model = trainer.model.model
    model.eval()
    torch.save(trainer.model.state_dict(), config.save_path)
    scaler = datamodule.dataset.scaler
    with open(save_dir / Path("scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)


def get_accelerator() -> Optional[str]:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return None
