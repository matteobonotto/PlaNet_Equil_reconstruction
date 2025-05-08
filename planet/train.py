# import os
# import sys
# sys.path.append(os.getcwd())

from typing import TypeAlias, Tuple
import random
import torch
from torch import Tensor
import lightning as L
from torch.utils.data import DataLoader, Subset
from multiprocessing import cpu_count

from .config import PlaNetConfig
from .model import PlaNetCore
from .loss import PlaNetLoss
from .data import PlaNetDataset, get_device


def collate_fun(batch: Tuple[Tuple[Tensor]]) -> Tuple[Tensor]:
    return (
        torch.stack([s[0] for s in batch], dim=0),  # measures
        torch.stack([s[1] for s in batch], dim=0),  # flux
        torch.stack([s[2] for s in batch], dim=0),  # rhs
        torch.stack([s[3] for s in batch], dim=0),  # RR
        torch.stack([s[4] for s in batch], dim=0),  # ZZ
        torch.stack([s[5] for s in batch], dim=0),  # L_ker
        torch.stack([s[6] for s in batch], dim=0),  # Dr_ker
    )

class DataModule(L.LightningDataModule):
    def __init__(self,config: PlaNetConfig):
        super().__init__()
        self.dataset = PlaNetDataset(
            path=config.dataset_path,
            is_physics_informed=config.is_physics_informed,
            nr=config.nr,
            nz=config.nz,
        )
        self.batch_size = config.batch_size
        self.num_workers = cpu_count()-2 if config.num_workers == -1 else config.num_workers
        self.split_dataset()
        
    def split_dataset(self, ratio:int = .1):
        idx = list(range(len(self.dataset)))
        idx_valid = random.sample(idx, k=int(ratio*len(idx)))
        idx_train = list(set(idx).difference(idx_valid))
        self.train_dataset = Subset(self.dataset, idx_train)
        self.val_dataset = Subset(self.dataset, idx_valid)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fun,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fun,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def setup(self, stage=None):
        pass


class LightningPlaNet(L.LightningModule):
    def __init__(self, config: PlaNetConfig):
        super().__init__()
        # device = get_device()
        self.model = PlaNetCore(
            hidden_dim=config.hidden_dim,
            nr=config.nr,
            nz=config.nz,
            branch_in_dim=config.branch_in_dim
        )
        self.loss_module = PlaNetLoss(is_physics_informed=config.is_physics_informed)

    def _compute_loss_batch(self, batch, batch_idx):
        measures, flux, rhs, RR, ZZ, L_ker, Df_ker = batch
        pred = self((measures, RR, ZZ))
        loss = self.loss_module(
            pred=pred,
            target=flux,
            rhs=rhs,
            Laplace_kernel=L_ker,
            Df_dr_kernel=Df_ker,
            RR=RR,
            ZZ=ZZ,
        )
        return loss

    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss_batch(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        # self.logger.log_metrics({'train_'+k:v for k,v in self.loss_module.log_dict.items()})
        for k, v in self.loss_module.log_dict.items():
            self.log(f"train_{k}", v, prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss_batch(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.001)

    # def on_fit_start(self):
    #     # Check if everything is correctly on device
    #     print("\nChecking model parameters devices:")
    #     for name, param in self.named_parameters():
    #         print(name, param.device)

    #     print("\nChecking model children modules devices:")
    #     for name, module in self.named_modules():
    #         if isinstance(module, torch.nn.Module):
    #             for pname, p in module.named_parameters(recurse=False):
    #                 print(f"{name}.{pname} -> {p.device}")

    #     pass
