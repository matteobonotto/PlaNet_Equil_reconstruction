import os
import sys

sys.path.append(os.getcwd())

import torch

from planet.loss import _compute_grad_shafranov_operator, Gauss_kernel
from planet.data import PlaNetDataset
from planet.train import DataModule
from planet.plot import fun_contourf_sol


def test_gs_operator():
    ###
    # ds = PlaNetDataset(path="planet/tests/data/iter_like_data_sample.h5")
    datamodule = DataModule(dataset_path="planet/tests/data/iter_like_data_sample.h5")
    dataloader = datamodule.train_dataloader()
    meas, flux, RHS_in, RR, ZZ, Laplace_kernel, Df_dr_kernel = next(iter(dataloader))

    ###
    gauss_kernel = torch.tensor(Gauss_kernel, dtype=torch.float32)
    rhs_computed = _compute_grad_shafranov_operator(
        flux, Laplace_kernel, Df_dr_kernel, RR, ZZ, gauss_kernel
    )

    x = rhs_computed[0, ...]
    fun_contourf_sol(
        rhs_computed[0, ...],
        RR[0, 1:-1, 1:-1],
        ZZ[0, 1:-1, 1:-1],
    )

    fun_contourf_sol(
        rhs_computed[0, ...] - RHS_in[0, ...],
        RR[0, 1:-1, 1:-1],
        ZZ[0, 1:-1, 1:-1],
    )


test_gs_operator()
