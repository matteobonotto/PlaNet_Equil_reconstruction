import os
import sys

sys.path.append(os.getcwd())

import torch

from planet.scripts.main_model import _compute_grad_shafranov_operator, Gauss_kernel
from planet.scripts.main_data import PlaNetDataset
from planet.scripts.main_train import DataModule


def test_gs_operator():
    ###
    ds = PlaNetDataset(path="planet_data_sample.h5")
    datamodule = DataModule(dataset_path="planet_data_sample.h5")
    dataloader = datamodule.train_dataloader()
    meas, flux, RHS_in, RR, ZZ, Laplace_kernel, Df_dr_kernel = next(iter(dataloader))

    ###
    Gauss_kernel = torch.tensor(Gauss_kernel, dtype=torch.float32)
    rhs_computed = _compute_grad_shafranov_operator(
        flux, Laplace_kernel, Df_dr_kernel, RR, ZZ, Gauss_kernel
    )


test_gs_operator()
