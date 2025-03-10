from copy import deepcopy
from typing import Tuple, List

import tensorflow as tf
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchinfo import summary

import numpy as np
import scipy
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
        self.trunk_r: List[Tuple[nn.Module, nn.Module]] = []
        self.trunk_z: List[Tuple[nn.Module, nn.Module]] = []
        channels: List[int] = [1, 8, 16, 32]
        for i in range(3):
            (in_channels, out_channels) = channels[i], channels[i + 1]
            self.trunk_r.append(
                (
                    Conv2dNornAct(in_channels=in_channels, out_channels=out_channels),
                    nn.MaxPool2d(kernel_size=2),
                )
            )
            self.trunk_z.append(
                (
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
        x_r = self.norm_r(x_r)
        for layer, maxpool2d in self.trunk_r:
            x_r = maxpool2d(layer(x_r))

        # branch for x_z
        x_z = self.norm_z(x_z)
        for layer, maxpool2d in self.trunk_z:
            x_z = maxpool2d(layer(x_z))

        # concatenate branches and output
        x = torch.cat((x_r, x_z), dim=1)  # [batch, 32+32, 4, 4] -> 1024
        x = self.flatten(x)
        x = self.act(self.linear_1(x))
        x = self.linear_2(x)
        return x


class BranchNet(nn.Module):
    def __init__(self, in_dim: int=302):
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
    def __init__(self, emb_dim:int = 2048):
        super().__init__()
        self.emb_dim = emb_dim
        self.linear = nn.Linear(in_features=64, out_features=emb_dim)
        self.act = TrainableSwish()
        self.decoder: List[Tuple[nn.Module, nn.Module]] = []
        # channels = [32, 16, 8, 4, 1]
        channels = [128, 32, 16, 8]
        for i in range(len(channels)-1):
            in_channels, out_channels = channels[i], channels[i + 1]
            self.decoder.append(
                (
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
        x = x_branch * x_trunk # [batch, 64]
        x = self.act(self.linear(x)) # [batch, 2048]
        x = x.reshape((-1, 128, 4, 4))

        for upsample, layer in self.decoder:
            x = layer(upsample(x))

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


if __name__ == "__main__":
    train_ds_1 = tf.data.Dataset.load(
        "./data/tf_Dataset_NeuralOpt_all_domain_only_32x32.data"
    )
    train_ds_2 = tf.data.Dataset.load(
        "./data/tf_Dataset_NeuralOpt_super_res_only_32x32.data"
    )
    # train_ds_1 = tf.data.Dataset.load('./gdrive/MyDrive/Colab_Notebooks/tf_Datasets/tf_Dataset_NeuralOpt_all_domain_only_64x64.data')
    # train_ds_2 = tf.data.Dataset.load('./gdrive/MyDrive/Colab_Notebooks/tf_Datasets/tf_Dataset_NeuralOpt_super_res_only_64x64.data')
    dict_geo = scipy.io.loadmat("./data/data_geo_Dataset_NeuralOpt_super_res_32x32.mat")

    train_ds = train_ds_1.concatenate(train_ds_2)
    train_ds = train_ds.shuffle(42)
    # train_ds = train_ds.batch(1024)

    x_ds, y_ds, RHS_in_ds, RR_ds, ZZ_ds, Laplace_kernel_ds, Df_dr_kernel_ds = iter(
        train_ds
    ).next()

    print(f"batch size: {x_ds.shape[0]}")
    print(train_ds.cardinality().numpy() * x_ds.shape[0])
    print(x_ds.shape)

    n_grid = RR_ds.shape[1]
    n_output = y_ds.shape[1]
    n_input = x_ds.shape[1]

    RR_pixels = dict_geo["RR_pixels"]
    ZZ_pixels = dict_geo["ZZ_pixels"]

    RHS_i = RHS_in_ds[-1, :, :].numpy()
    psi_i = y_ds[-1, :, :].numpy()

    inputs = [x_ds, RR_ds, ZZ_ds]

    branch_net = BranchNet(in_dim=x_ds.shape[-1])
    out_branch = branch_net(torch.tensor(x_ds.numpy()))

    trunk_net = TrunkNet()
    out_trunk = trunk_net(
        torch.tensor(RR_ds.numpy()).unsqueeze(1),
        torch.tensor(ZZ_ds.numpy()).unsqueeze(1),
    )

    decoder = Decoder()
    out = decoder(out_trunk, out_branch)

    planet = PlaNetCore()
    summary(planet, input_data=(
        torch.tensor(x_ds.numpy()), 
        torch.tensor(RR_ds.numpy()).unsqueeze(1), 
        torch.tensor(ZZ_ds.numpy()).unsqueeze(1)
    ))
