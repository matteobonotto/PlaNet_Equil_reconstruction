


from typing import Dict, List, Tuple
import torch
from torch import Tensor, nn
import numpy as np
import torch.nn.functional as F





DTYPE = torch.float32

# Gauss_tensor = tf.expand_dims(
#     tf.expand_dims(Gaussian_kernel[::-1, ::-1], axis=-1), axis=-1
# )

# Gauss_tensor = torch.tensor(Gauss_tensor, dtype=DTYPE)


class TrainableSwish(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x: Tensor) -> Tensor:
        return swish(x, self.beta)


def swish(x: Tensor, beta: float = 1.0) -> Tensor:
    return x * F.sigmoid(beta * x)


class LineardNornAct(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.norm = nn.BatchNorm2d(num_features=out_features)
        self.act = TrainableSwish()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.linear(x)))


class MLPStack(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_layers: int = 3, hidden_dim: int = 128):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                in_features, out_features = in_dim, hidden_dim
            elif i == n_layers -1:
                in_features, out_features = hidden_dim, out_dim
            else:
                in_features, out_features = hidden_dim, hidden_dim
            self.layers.append(
                LineardNornAct(in_features=in_features, out_features=out_features)
            )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class PlaNetCoreMLP(nn.Module):
    def __init__(
        self,
        meas_dim: int = 302,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ):
        super().__init__()
        self.trunk = MLPStack(in_dim=2, out_dim=hidden_dim, n_layers=n_layers)
        self.branch = MLPStack(in_dim=meas_dim, out_dim=hidden_dim, n_layers=n_layers)
        self.decoder = MLPStack(in_dim=hidden_dim, out_dim=1, n_layers=n_layers)

    def forward(self, x: Tensor) -> Tensor:
        x_meas, rz = x[:, :-2], x[:, -2:]
        out_branch = self.branch(x_meas)
        out_trunk = self.trunk(rz)
        return self.decoder(out_branch * out_trunk)
    

    

def compute_partial_derivative(f: Tensor, var: Tensor) -> Tensor:
    d = torch.autograd.grad(f, var, grad_outputs=torch.ones_like(f), create_graph=True)[0]
    return d

def compute_gso(pred: Tensor, rz: Tensor) -> Tensor:
    # Create coordinates with requires_grad=True
    r = rz[..., -2]
    z = rz[..., -1]

    # First-order derivatives
    # df_dr = torch.autograd.grad(pred, r, grad_outputs=torch.ones_like(pred), create_graph=True)[0]
    # df_dz = torch.autograd.grad(pred, z, grad_outputs=torch.ones_like(pred), create_graph=True)[0]

    # # Second-order (for Laplacian)
    # d2f_dr2 = torch.autograd.grad(df_dr, r, grad_outputs=torch.ones_like(df_dr), create_graph=True)[0]
    # d2f_dz2 = torch.autograd.grad(df_dz, z, grad_outputs=torch.ones_like(df_dz), create_graph=True)[0]

    df_dr = compute_partial_derivative(pred, r)
    df_dz = compute_partial_derivative(pred, z)
    d2f_dr2 = compute_partial_derivative(df_dr, r)
    d2f_dz2 = compute_partial_derivative(df_dz, z)

    gso = d2f_dr2 + d2f_dz2 - df_dr / r
    return gso

class GSOperatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred: Tensor,
        rhs: Tensor,
        rz: Tensor,
    ) -> Tensor:

        gso = compute_gso(pred=pred, rz=rz)

        return self.mse(gso, rhs)


MAP_PDELOSS: Dict[str, nn.Module] = {"grad_shafranov_operator": GSOperatorLoss}


class PlaNetLoss(nn.Module):
    log_dict: Dict[str, float] = {}

    def __init__(
        self,
        is_physics_informed: bool = True,
        scale_mse: float = 1.0,
        scale_pde: float = 0.1,  # for better stability
        pde_loss_class: str = "grad_shafranov_operator",
    ):
        super().__init__()
        self.is_physics_informed = is_physics_informed
        self.loss_mse = nn.MSELoss()
        self.loss_pde = MAP_PDELOSS[pde_loss_class]()
        self.scale_mse = scale_mse
        self.scale_pde = scale_pde

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        rz: Tensor,
        rhs: Tensor,
    ) -> Tensor:
        mse_loss = self.scale_mse * self.loss_mse(input=pred, target=target)
        self.log_dict["mse_loss"] = mse_loss.item()
        if not self.is_physics_informed:
            return mse_loss
        else:
            pde_loss = self.scale_pde * self.loss_pde(
                pred=pred,
                rhs=rhs, 
                rz=rz,
            )
            self.log_dict["pde_loss"] = pde_loss.item()
            return mse_loss + pde_loss





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
        do_super_resolution: bool = False,
    ):
        self.dtype = dtype
        self.device = get_device()
        self.scaler = StandardScaler()
        self.is_physics_informed = is_physics_informed
        self.nr = nr
        self.nz = nz
        self.do_super_resolution = do_super_resolution

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

        if self.nr != self.RR.shape[0] or self.nz != self.RR.shape[1]:
            rr = np.linspace(self.RR[0, 0], self.RR[0, -1], self.nr)
            zz = np.linspace(self.ZZ[0, 0], self.RR[-1, 0], self.nr)
            self.RR, self.ZZ = np.meshgrid(rr, zz)
            self.base_RR, self.base_ZZ = data["RR_grid"], data["ZZ_grid"]

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

        if flux.shape[1] != RR.shape[0] or flux.shape[1] != RR.shape[1]:
            flux = interp_fun(
                f=flux, RR=self.base_RR, ZZ=self.base_ZZ, rr=self.RR, zz=self.ZZ
            )
            rhs = interp_fun(
                f=rhs, RR=self.base_RR, ZZ=self.base_ZZ, rr=self.RR, zz=self.ZZ
            )

        if random.random() > 0.5 and self.do_super_resolution:
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
    def __init__(self, config: PlaNetConfig):
        super().__init__()
        self.dataset = PlaNetDataset(
            path=config.dataset_path,
            is_physics_informed=config.is_physics_informed,
            nr=config.nr,
            nz=config.nz,
            do_super_resolution=config.do_super_resolution,
        )
        self.batch_size = config.batch_size
        self.num_workers = (
            cpu_count() - 2 if config.num_workers == -1 else config.num_workers
        )
        self.split_dataset()

    def split_dataset(self, ratio: int = 0.1):
        idx = list(range(len(self.dataset)))
        idx_valid = random.sample(idx, k=int(ratio * len(idx)))
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
            branch_in_dim=config.branch_in_dim,
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