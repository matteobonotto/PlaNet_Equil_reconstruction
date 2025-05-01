import os
import sys

sys.path.append(os.getcwd())
import torch
import lightning as L
from torch.utils.data import DataLoader

from .model import PlaNetCore, PlaNetLoss
from .data import PlaNetDataset, get_device


class DataModule(L.LightningDataModule):
    def __init__(self, dataset_path: str):
        super().__init__()
        self.train_dataset = PlaNetDataset(path=dataset_path)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=64, shuffle=True)

    def setup(self, stage=None):
        pass


class LightningPlaNet(L.LightningModule):
    def __init__(self):
        super().__init__()
        # device = get_device()
        self.model = PlaNetCore()
        self.loss_module = PlaNetLoss()

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
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.001)

    def on_fit_start(self):
        # Check if everything is correctly on device
        print("\nChecking model parameters devices:")
        for name, param in self.named_parameters():
            print(name, param.device)

        print("\nChecking model children modules devices:")
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Module):
                for pname, p in module.named_parameters(recurse=False):
                    print(f"{name}.{pname} -> {p.device}")

        pass
