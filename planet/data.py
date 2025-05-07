import random
import numpy as np
from numpy import ndarray
from functools import partial
from typing import Optional, Tuple, Any, List
import torch
from torch import Tensor
from sklearn.preprocessing import StandardScaler

import h5py
from scipy.interpolate import RegularGridInterpolator


from planet.constants import RANDOM_SEED

random.seed(RANDOM_SEED)


def sample_random_subgrids(
    RR_min: float,
    RR_max: float,
    ZZ_min: float,
    ZZ_max: float,
    nr: int,
    nz: int,
    seed: Optional[int],
) -> Tuple[ndarray, ndarray]:
    delta_r_min = (RR_max - RR_min) / 3
    delta_r_max = RR_max - RR_min

    delta_z_min = (ZZ_max - ZZ_min) / 6
    delta_z_max = ZZ_max - ZZ_min

    if seed is not None:
        np.random.seed(seed)
    delta_r = np.random.uniform(delta_r_min, delta_r_max, 1)
    r0 = np.random.uniform(RR_min, RR_min + delta_r_max - delta_r, 1)

    delta_z = np.random.uniform(delta_z_min, delta_z_max, 1)
    z0 = np.random.uniform(ZZ_min, ZZ_min + delta_z_max - delta_z, 1)

    rr = np.linspace(r0, r0 + delta_r, nr)
    zz = np.linspace(z0, z0 + delta_z, nz)

    rr_grid, zz_grid = np.meshgrid(rr, zz, indexing="xy")

    return (rr_grid, zz_grid)


def get_box_from_grid(rr_grid: ndarray, zz_grid: ndarray) -> ndarray:
    return np.array(
        [
            [rr_grid.min(), zz_grid.min()],
            [rr_grid.max(), zz_grid.min()],
            [rr_grid.max(), zz_grid.max()],
            [rr_grid.min(), zz_grid.max()],
            [rr_grid.min(), zz_grid.min()],
        ]
    )


def interp_fun(
    f: ndarray, RR: ndarray, ZZ: ndarray, rr: ndarray, zz: ndarray
) -> ndarray:
    x_pts = RR[0, :].ravel()
    y_pts = ZZ[:, 0].ravel()
    interp_func = RegularGridInterpolator((x_pts, y_pts), f.T)
    f_int = interp_func(
        np.column_stack(
            (
                rr.reshape(-1, 1),
                zz.reshape(-1, 1),
            )
        ),
        method="quintic",
    ).reshape(rr.shape)
    return f_int


def compute_Grda_Shafranov_kernels(RR: ndarray, ZZ: ndarray) -> Tuple[ndarray, ndarray]:
    hr = RR[1, 2] - RR[1, 1]
    hz = ZZ[2, 1] - ZZ[1, 1]
    alfa = -2 * (hr**2 + hz**2)
    Laplace_kernel = np.array(
        ([0, hr**2 / alfa, 0], [hz**2 / alfa, 1, hz**2 / alfa], [0, hr**2 / alfa, 0])
    )
    Df_dr_kernel = (
        np.array(([0, 0, 0], [+1, 0, -1], [0, 0, 0]))
        / (2 * hr * alfa)
        * (hr**2 * hz**2)
    )
    return Laplace_kernel, Df_dr_kernel


def _to_tensor(
    device: torch.device, inputs: Tuple[Any], dtype: torch.dtype
) -> Tuple[Tensor]:
    inputs_t: List[Tensor] = []
    for x in inputs:
        inputs_t.append(
            torch.tensor(
                x,
                dtype=dtype,
                # device=device,
            )
        )
    return tuple(inputs_t)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class PlaNetDataset:
    def __init__(
        self,
        path: str,
        dtype: torch.dtype = torch.float32,
        is_physics_informed: bool = True,
        nr: int = 64,
        nz: int = 64,
    ):
        self.dtype = dtype
        self.device = get_device()
        self.scaler = StandardScaler()
        self.is_physics_informed = is_physics_informed
        self.nr = nr
        self.nz = nz

        data = read_h5_numpy(path)
        self.inputs = self.scaler.fit_transform(
            np.column_stack(
                (data["measures"], data["coils_current"], data["p_profile"])
            )
        )

        self.flux = data["flux"]
        self.rhs = data["rhs"]
        self.RR = data["RR_grid"]
        self.ZZ = data["ZZ_grid"]

        self.sample_random_subgrids = partial(
            sample_random_subgrids,
            RR_min=self.RR.min(),
            RR_max=self.RR.max(),
            ZZ_min=self.ZZ.min(),
            ZZ_max=self.ZZ.max(),
            nr=self.RR.shape[0],
            nz=self.RR.shape[1],
            seed=RANDOM_SEED,
        )

    def get_scaler(self) -> StandardScaler:
        return self.scaler

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor]:
        inputs = self.inputs[idx, ...]
        flux = self.flux[idx, ...]
        rhs = self.rhs[idx, ...]
        RR = self.RR
        ZZ = self.ZZ

        if random.random() > 0.5:
            # interpolate on a subgrid
            rr, zz = self.sample_random_subgrids()
            flux = interp_fun(f=flux, RR=self.RR, ZZ=self.ZZ, rr=rr, zz=zz)
            if self.is_physics_informed:
                rhs = interp_fun(
                    f=rhs,
                    RR=self.RR,
                    ZZ=self.ZZ,
                    rr=rr[1:-1, 1:-1],
                    zz=zz[1:-1, 1:-1],
                )
            else:
                rhs = np.zeros_like(rhs[1:-1, 1:-1])
            RR = rr
            ZZ = zz
        else:
            rhs = rhs[1:-1, 1:-1]

        L_ker, Df_ker = compute_Grda_Shafranov_kernels(RR=RR, ZZ=ZZ)

        return _to_tensor(
            device=self.device,
            dtype=self.dtype,
            inputs=(inputs, flux, rhs, RR, ZZ, L_ker, Df_ker),
        )


def write_h5(
    data: dict,
    filename: str,
    dtype: str = "float64",
    # compression : str = 'lzf',
    # compression_opts : int = 1,
    # verbose : bool = False,
):

    compression: str = "lzf"
    compression: int = 1

    kwargs = {
        "dtype": dtype,
        "compression": compression,
    }

    # t_start = time.time()
    with h5py.File(filename + ".h5", "w") as hf:
        for key, item in data.items():
            hf.create_dataset(key, data=item, shape=item.shape, **kwargs)
    hf.close()


def read_h5_numpy(
    filename: str,
):
    data = {}
    with h5py.File(filename, "r") as hf:
        for key, item in hf.items():
            data.update({key: item[()]})
    return data
