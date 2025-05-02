from copy import deepcopy
from typing import Tuple, List

import tensorflow as tf
import torch
from torch import Tensor, nn
import torch.nn.functional as F

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines


def fun_contourf_sol(z, RR, ZZ):
    plt.figure()
    plt.contourf(RR, ZZ, z, 20)
    plt.axis("equal")
    plt.colorbar()
    plt.show()
    return


def fun_contour_sol(z, RR, ZZ):
    plt.figure()
    plt.contour(RR, ZZ, z, 20)
    plt.axis("equal")
    plt.colorbar()
    plt.show()
    return


def fun_contour_compare_sol(z_ref, z, RR, ZZ):
    l1 = mlines.Line2D([], [], label="DNN")
    l2 = mlines.Line2D([], [], color="black", label="FRIDA")

    plt.figure()
    plt.contour(RR, ZZ, z, 10)
    plt.colorbar()
    plt.contour(RR, ZZ, z_ref, 10, colors="black", linestyles="dashed")
    plt.legend(handles=[l1, l2])
    plt.axis("equal")
    plt.show()
    return


DTYPE = torch.float32
Gauss_kernel = np.array(([1, 2, 1], [2, 4, 2], [1, 2, 1])) / (16)
# Gauss_tensor = tf.expand_dims(
#     tf.expand_dims(Gaussian_kernel[::-1, ::-1], axis=-1), axis=-1
# )

# Gauss_tensor = torch.tensor(Gauss_tensor, dtype=DTYPE)


def fun_GSoperator_NN_conv_smooth_batch_adaptive(
    f,
    Laplace_kernel_ds,
    Df_dr_kernel_ds,
    RR_ds,
    ZZ_ds,
):

    f = tf.transpose(f, [3, 1, 2, 0])
    Lpsi = tf.nn.depthwise_conv2d(
        f,
        tf.transpose(tf.expand_dims(Laplace_kernel_ds, axis=-1), [1, 2, 0, 3]),
        strides=[1, 1, 1, 1],
        padding="VALID",
    )
    Lpsi = tf.transpose(
        Lpsi, [3, 1, 2, 0]
    )  # no need to be transposed becaused Laplacian filter is left/rigth symmetric
    Dpsi_dr = tf.nn.depthwise_conv2d(
        f,
        tf.transpose(tf.expand_dims(Df_dr_kernel_ds, axis=-1), [1, 2, 0, 3]),
        strides=[1, 1, 1, 1],
        padding="VALID",
    )
    Dpsi_dr = (
        -Dpsi_dr
    )  # necessary because nn.depthwise_conv2d filters has to be transposed to perform real convolution (here [+h 0 -h] -> [-h 0 +h])
    Dpsi_dr = tf.transpose(Dpsi_dr, [3, 1, 2, 0])
    RR_in = tf.expand_dims(RR_ds[:, 1:-1, 1:-1], axis=-1)
    Dpsi_dr = tf.math.divide(Dpsi_dr, RR_in)

    GS_ope = Lpsi - Dpsi_dr

    hr = RR_ds[:, 1, 2] - RR_ds[:, 1, 1]
    hz = ZZ_ds[:, 2, 1] - ZZ_ds[:, 1, 1]
    alfa = -2 * (hr**2 + hz**2)

    hr = tf.expand_dims(tf.expand_dims(tf.expand_dims(hr, axis=-1), axis=-1), axis=-1)
    hz = tf.expand_dims(tf.expand_dims(tf.expand_dims(hz, axis=-1), axis=-1), axis=-1)
    alfa = tf.expand_dims(
        tf.expand_dims(tf.expand_dims(alfa, axis=-1), axis=-1), axis=-1
    )

    GS_ope = GS_ope * alfa / (hr**2 * hz**2)
    # GS_ope[0,:10,0]

    GS_ope = tf.nn.conv2d(GS_ope, Gauss_tensor, strides=[1, 1, 1, 1], padding="SAME")
    GS_ope = tf.squeeze(GS_ope, axis=-1)

    # GS_ope_padded = RHS_ds.numpy()
    # GS_ope_padded[:,1:-1,1:-1] = GS_ope
    # GS_ope = tf.constant(GS_ope_padded)
    return GS_ope


def loss_fun_MSE(y_ds, predictions):
    loss_MSE = tf.reduce_mean(tf.square(y_ds - tf.squeeze(predictions)))
    return loss_MSE


def loss_fun_PDE_adaptive(
    predictions, RHS_in_ds, Laplace_kernel_ds, Df_dr_kernel_ds, RR_ds, ZZ_ds
):
    GS_ope_ref = RHS_in_ds
    GS_ope_ds = fun_GSoperator_NN_conv_smooth_batch_adaptive(
        predictions, Laplace_kernel_ds, Df_dr_kernel_ds, RR_ds, ZZ_ds
    )
    loss_PDE = tf.reduce_mean(tf.square(GS_ope_ref - GS_ope_ds))
    return 0.1 * loss_PDE


import torch.nn.functional as F


def _compute_grad_shafranov_operator(
    pred: Tensor,
    Laplace_kernel: Tensor,
    Df_dr_kernel: Tensor,
    RR: Tensor,
    ZZ: Tensor,
    Gauss_kernel: Tensor,
) -> Tensor:
    """
    This implementation is taken from the paper https://doi.org/10.1016/j.fusengdes.2024.114193,
    cited as [1] in the code
    """

    batch_size = pred.shape[0]

    # Compute the Laplace and the Dpsi_dr operators
    Lpsi = F.conv2d(
        pred[:, None, ...].permute(1, 0, 2, 3),
        weight=Laplace_kernel[:, None, ...],
        groups=batch_size,
    ).permute(1, 0, 2, 3)

    # The '-' is necessary because the depthwise conv filtershas to be transposed
    # to perform real convolution (here [+h 0 -h] -> [-h 0 +h])
    Dpsi_dr = -F.conv2d(
        pred[:, None, ...].permute(1, 0, 2, 3),
        weight=Df_dr_kernel[:, None, ...],
        groups=batch_size,
    ).permute(1, 0, 2, 3)
    Dpsi_dr = torch.div(Dpsi_dr, RR[:, None, 1:-1, 1:-1])

    # the LHS in equation 12 of [1]
    lhs = Lpsi - Dpsi_dr

    # GS operator (RHS of requation 12 of [1])
    hr = (RR[:, 1, 2] - RR[:, 1, 1])[:, None, None, None]
    hz = (ZZ[:, 2, 1] - ZZ[:, 1, 1])[:, None, None, None]
    alfa = -2 * (hr**2 + hz**2)
    GS_ope = lhs * alfa / (hr**2 * hz**2)

    # convolve with the gaussian kernel to smooth the solution a bit
    GS_ope_smooth = -F.conv2d(
        GS_ope,
        weight=Gauss_kernel[None, None, ...],
        groups=1,
        padding="same",
    )

    """
    ### this is the scipy implementation
    hr = (RR[0,1,2] - RR[0,1,1]).cpu().detach().numpy()
    hz = (ZZ[0,2,1] - ZZ[0,1,1]).cpu().detach().numpy()
    kr = np.array(([0, 0, 0], [1, -2, 1], [0, 0, 0]))*hz**2
    kz = np.transpose(np.array(([0, 0, 0], [1, -2, 1], [0, 0, 0])))*hr**2

    L_kernel = (kr + kz)/(hr**2*hz**2)

    Df_dr_kernel = np.array(([0, 0, 0], [+1, 0, -1], [0, 0, 0]))/(2*hr)

    Lpsi = signal.convolve2d(psi_i, Laplace_kernel, mode='valid')
    # Lpsi = Lpsi/(hr**2*hz**2)
    Dpsi_dr = signal.convolve2d(psi_i, Df_dr_kernel, mode='valid')
    # LHS_conv = Lpsi - Dpsi_dr/RR_in
    GS_ope = Lpsi - Dpsi_dr/RR_in

    ### this is the original tensorflow implementation
    # TF tensors arrives with the shape [batch, H, W, channel]
    pred_tf = pred[:, :, :, None]
    f = tf.expand_dims(tf.constant(pred.detach().cpu().numpy()), axis=-1)
    Laplace_kernel_ds = tf.expand_dims(
        tf.constant(Laplace_kernel.detach().cpu().numpy()), axis=-1
    )
    Df_dr_kernel_ds = tf.expand_dims(
        tf.constant(Df_dr_kernel.detach().cpu().numpy()), axis=-1
    )
    RR_ds = tf.expand_dims(tf.constant(RR.detach().cpu().numpy()), axis=-1)
    ZZ_ds = tf.expand_dims(tf.constant(ZZ.detach().cpu().numpy()), axis=-1)
    

    # f = tf.transpose(f, [3, 1, 2, 0])
    Lpsi = tf.nn.depthwise_conv2d(
        f,
        tf.transpose(tf.expand_dims(Laplace_kernel_ds, axis=-1), [1, 2, 0, 3]),
        strides=[1, 1, 1, 1],
        padding="VALID",
    )
    Lpsi = tf.transpose(
        Lpsi, [3, 1, 2, 0]
    )  # no need to be transposed becaused Laplacian filter is left/rigth symmetric
    Dpsi_dr = tf.nn.depthwise_conv2d(
        f,
        tf.transpose(tf.expand_dims(Df_dr_kernel_ds, axis=-1), [1, 2, 0, 3]),
        strides=[1, 1, 1, 1],
        padding="VALID",
    )
    Dpsi_dr = (
        -Dpsi_dr
    )  # necessary because nn.depthwise_conv2d filters has to be transposed to perform real convolution (here [+h 0 -h] -> [-h 0 +h])
    Dpsi_dr = tf.transpose(Dpsi_dr, [3, 1, 2, 0])
    RR_in = tf.expand_dims(RR_ds[:, 1:-1, 1:-1], axis=-1)
    Dpsi_dr = tf.math.divide(Dpsi_dr, RR_in)

    GS_ope = Lpsi - Dpsi_dr

    hr = RR_ds[:, 1, 2] - RR_ds[:, 1, 1]
    hz = ZZ_ds[:, 2, 1] - ZZ_ds[:, 1, 1]
    alfa = -2 * (hr**2 + hz**2)

    hr = tf.expand_dims(tf.expand_dims(tf.expand_dims(hr, axis=-1), axis=-1), axis=-1)
    hz = tf.expand_dims(tf.expand_dims(tf.expand_dims(hz, axis=-1), axis=-1), axis=-1)
    alfa = tf.expand_dims(
        tf.expand_dims(tf.expand_dims(alfa, axis=-1), axis=-1), axis=-1
    )

    GS_ope = GS_ope * alfa / (hr**2 * hz**2)
    # GS_ope[0,:10,0]

    GS_ope = tf.nn.conv2d(GS_ope, Gauss_tensor, strides=[1, 1, 1, 1], padding="SAME")
    GS_ope = tf.squeeze(GS_ope, axis=-1)
    
    """
    # it appears that there is a - sign missing. Check out why!
    return -GS_ope_smooth.squeeze()


class TrainableSwish(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x: Tensor) -> Tensor:
        return swish(x, self.beta)


def swish(x: Tensor, beta: float = 1.0) -> Tensor:
    return x * F.sigmoid(beta * x)


class Conv2dNornAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        padding: str = "same",
    ):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.act = TrainableSwish()

    def forward(self, x: Tensor) -> Tensor:
        """
        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            dtype=DTYPE,
        )(x)
        x = layers.BatchNormalization()(x)
        x = Swish(beta=1.0, trainable=True)(x)
        """
        return self.act(self.norm(self.conv2d(x)))


class TrunkNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_r = nn.BatchNorm2d(num_features=1)
        self.norm_z = nn.BatchNorm2d(num_features=1)
        self.trunk_r = nn.ModuleList()
        self.trunk_z = nn.ModuleList()
        channels: List[int] = [1, 8, 16, 32]
        for i in range(3):
            (in_channels, out_channels) = channels[i], channels[i + 1]
            self.trunk_r.append(
                nn.Sequential(
                    Conv2dNornAct(in_channels=in_channels, out_channels=out_channels),
                    nn.MaxPool2d(kernel_size=2),
                )
            )
            self.trunk_z.append(
                nn.Sequential(
                    Conv2dNornAct(in_channels=in_channels, out_channels=out_channels),
                    nn.MaxPool2d(kernel_size=2),
                )
            )
        # [batch, 1, 32, 32]
        # [batch, 8, 16, 16]
        # [batch, 16, 8, 8]
        # [batch, 32, 4, 4] -> 512
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(in_features=1024, out_features=128)
        self.act = TrainableSwish()
        self.linear_2 = nn.Linear(in_features=128, out_features=64)

    def forward(self, x_r: Tensor, x_z: Tensor) -> Tensor:
        """
        # x_r = input_query_RR
        x_r = layers.BatchNormalization()(input_query_RR)
        for i in range(3):
            x_r = conv2D_Norm_activation(x_r, filters=(i + 1) * 8, kernel_size=(3, 3))
            x_r = layers.MaxPooling2D(pool_size=(2, 2))(x_r)

        # x_z = input_query_ZZ
        x_z = layers.BatchNormalization()(input_query_ZZ)
        for i in range(3):
            x_z = conv2D_Norm_activation(x_z, filters=(i + 1) * 8, kernel_size=(3, 3))
            x_z = layers.MaxPooling2D(pool_size=(2, 2))(x_z)

        out_trunk = layers.Concatenate()([x_r, x_z])
        out_trunk = layers.Flatten()(out_trunk)
        out_trunk = layers.Dense(128, dtype=DTYPE)(out_trunk)
        out_trunk = Swish(beta=1.0, trainable=True)(out_trunk)

        for i in range(2):
            out_trunk = layers.Dense(64, dtype=DTYPE)(out_trunk)
        out_trunk = Swish(beta=1.0, trainable=True)(out_trunk)
        """

        # branch for x_r
        x_r = self.norm_r(x_r.unsqueeze(1))
        for layer in self.trunk_r:
            x_r = layer(x_r)

        # branch for x_z
        x_z = self.norm_z(x_z.unsqueeze(1))
        for layer in self.trunk_z:
            x_z = layer(x_z)

        # concatenate branches and output
        x = torch.cat((x_r, x_z), dim=1)  # [batch, 32+32, 4, 4] -> 1024
        x = self.flatten(x)
        x = self.act(self.linear_1(x))
        x = self.linear_2(x)
        return x


class BranchNet(nn.Module):
    def __init__(self, in_dim: int = 302):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_dim, out_features=256)
        self.norm_1 = nn.BatchNorm1d(num_features=256)
        self.act_1 = TrainableSwish(beta=1.0)
        self.linear_2 = nn.Linear(in_features=256, out_features=128)
        self.norm_2 = nn.BatchNorm1d(num_features=128)
        self.act_2 = TrainableSwish(beta=1.0)
        self.linear_3 = nn.Linear(in_features=128, out_features=64)

    def forward(self, x: Tensor) -> Tensor:
        """
        x = layers.Dense(256, dtype=DTYPE)(input_fun)
        x = Swish(beta=1.0, trainable=True)(x)

        x = layers.Dense(128, dtype=DTYPE)(x)
        x = Swish(beta=1.0, trainable=True)(x)

        x = layers.Dense(64, dtype=DTYPE)(x)
        out_branch = Swish(beta=1.0, trainable=True)(x)
        """
        x = self.act_1(self.norm_1(self.linear_1(x)))
        x = self.act_2(self.norm_2(self.linear_2(x)))
        x = self.linear_3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, emb_dim: int = 2048):
        super().__init__()
        self.emb_dim = emb_dim
        self.linear = nn.Linear(in_features=64, out_features=emb_dim)
        self.act = TrainableSwish()
        self.decoder = nn.ModuleList()
        # channels = [32, 16, 8, 4, 1]
        channels = [128, 32, 16, 8]
        for i in range(len(channels) - 1):
            in_channels, out_channels = channels[i], channels[i + 1]
            self.decoder.append(
                nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    Conv2dNornAct(in_channels=in_channels, out_channels=out_channels),
                )
            )
        self.conv = nn.Conv2d(
            in_channels=channels[-1], out_channels=1, kernel_size=(1, 1), padding="same"
        )

    def forward(self, x_trunk: Tensor, x_branch: Tensor) -> Tensor:
        """
        # Multiply layer
        out_multiply = layers.Multiply(name="Multiply")([out_branch, out_trunk])

        # conv2d-based decoder
        x_dec = layers.Dense(
            neuron_FC,
            dtype=DTYPE,
        )(out_multiply)
        x_dec = Swish(beta=1.0, trainable=True)(x_dec)

        x_dec = layers.Reshape(target_shape=(n_w, n_h, n_c))(x_dec)

        x_dec = layers.UpSampling2D(size=(2, 2), interpolation=interpolation)(x_dec)
        x_dec = conv2D_Norm_activation(x_dec, filters=32, kernel_size=(3, 3))

        x_dec = layers.UpSampling2D(size=(2, 2), interpolation=interpolation)(x_dec)
        x_dec = conv2D_Norm_activation(x_dec, filters=16, kernel_size=(3, 3))

        x_dec = layers.UpSampling2D(size=(2, 2), interpolation=interpolation)(x_dec)
        x_dec = conv2D_Norm_activation(x_dec, filters=8, kernel_size=(3, 3))

        out_grid = layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            activation="linear",
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            dtype=DTYPE,
        )(x_dec)

        outputs = out_grid
        """
        # batch, n_hidden = x_branch.shape
        x = x_branch * x_trunk  # [batch, 64]
        x = self.act(self.linear(x))  # [batch, 2048]
        x = x.reshape((-1, 128, 4, 4))

        for layer in self.decoder:
            x = layer(x)

        x = self.conv(x).squeeze()
        return x


class PlaNetCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = TrunkNet()
        self.branch = BranchNet()
        self.decoder = Decoder()

    def forward(self, x: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        x_meas, x_r, x_z = x
        out_branch = self.branch(x_meas)
        out_trunk = self.trunk(x_r, x_z)
        return self.decoder(out_branch, out_trunk)


class PDELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.register_buffer(
            "Gauss_kernel", torch.tensor(Gauss_kernel, dtype=torch.float32)
        )

    def forward(
        self,
        pred: Tensor,
        rhs: Tensor,
        Laplace_kernel: Tensor,
        Df_dr_kernel: Tensor,
        RR: Tensor,
        ZZ: Tensor,
    ) -> Tensor:
        rhs_computed = _compute_grad_shafranov_operator(
            pred, Laplace_kernel, Df_dr_kernel, RR, ZZ, self.Gauss_kernel
        )
        return self.mse(rhs_computed, rhs)


class PlaNetLoss(nn.Module):
    def __init__(self, scale_mse: float = 1.0, scale_pde: float = 1.0):
        super().__init__()
        self.loss_mse = nn.MSELoss()
        self.loss_pde = PDELoss()
        self.scale_mse = scale_mse
        self.scale_pde = scale_pde

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        rhs: Tensor,
        Laplace_kernel: Tensor,
        Df_dr_kernel: Tensor,
        RR: Tensor,
        ZZ: Tensor,
    ) -> Tensor:
        mse_loss = self.scale_mse * self.loss_mse(input=pred, target=target)
        pde_loss = self.scale_pde * self.loss_pde(
            pred=pred,
            rhs=rhs,
            Laplace_kernel=Laplace_kernel,
            Df_dr_kernel=Df_dr_kernel,
            RR=RR,
            ZZ=ZZ,
        )
        return mse_loss + pde_loss
