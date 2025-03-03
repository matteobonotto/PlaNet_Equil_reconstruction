# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 12:20:13 2022

@author: bonotto
"""

# %%

"""
###############################################################################
###############################################################################
"""


### CLEAR WRKSPACE
def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == "_":
            continue
        if "func" in str(globals()[var]):
            continue
        if "module" in str(globals()[var]):
            continue

        del globals()[var]


if __name__ == "__main__":
    clear_all()
    # insert here your code

clear_all
"""
###############################################################################
###############################################################################
"""


###############################################################################

import os

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Helper libraries
import numpy as np
import pandas as pd
from numpy import matlib as mb
import matplotlib.pyplot as plt

import scipy.io

# !pip install seaborn
import seaborn as sns

sns.set_style("darkgrid")

# !pip install seaborn

import random

import matplotlib.lines as mlines


# pip install mat4py

# Load the TensorBoard notebook extension.
# %load_ext tensorboard

from datetime import datetime
from packaging import version

# !pip install --upgrade --force-reinstall tensorflow
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# import tensorflow_addons as tfa

# !pip install tensorflow
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential


print("TensorFlow version: ", tf.__version__)
assert (
    version.parse(tf.__version__).release[0] >= 2
), "This notebook requires TensorFlow 2.0 or above."

# import tensorboard
# tensorboard.__version__
# Clear any logs from previous runs
# !rm -rf ./logs/

# C:\Users\bonotto\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\python.exe -m pip install --upgrade pip'

# !pip install sklearn
# !pip install --upgrade --force-reinstall scikeras


# from scikeras.wrappers import KerasRegressor

from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin
from sklearn.metrics import (
    r2_score,
    f1_score,
    accuracy_score,
    roc_curve,
    roc_auc_score,
    mean_squared_error,
)


# from sklearn.externals import joblib
import scipy.interpolate as interp

from sklearn.metrics import r2_score

from matplotlib import path

from mpl_toolkits import mplot3d

import matplotlib.tri as tri
import matplotlib.cm as cm

# !pip install --upgrade --force-reinstall scipy
from scipy import interpolate
from scipy import signal
import scipy.ndimage
import time

import mat73


# import h5py
# f = h5py.File(filename, 'r')

"""
###############################################################################
### Load database 
"""

try:  # Dell XPS
    folder = os.path.join(
        r"C:\Users\matte\Dropbox\PhD\RESEARCH_ACTIVITY\ML\Equilibrium_ML\database_equil_ITER_FRIDA"
    )
    ### load probes data and fluxes
    filename = os.path.join(
        folder + r"\Database_psi_rec_ConvNet_ITER_TensorFlow_v4_15000.mat"
    )
    ind_case = 15000
    filename = os.path.join(
        folder + r"/Database_psi_rec_ConvNet_ITER_TensorFlow_v4.mat"
    )
    mat = mat73.loadmat(filename)
except:
    try:  # desktop RFX
        folder = os.path.join(
            r"C:\Users\bonotto\Dropbox\PhD\RESEARCH_ACTIVITY\ML\Equilibrium_ML\database_equil_ITER_FRIDA"
        )
        ### load probes data and fluxes
        filename = os.path.join(
            folder + r"\Database_psi_rec_ConvNet_ITER_TensorFlow_v4_15000.mat"
        )
        ind_case = 15000
        filename = os.path.join(
            folder + r"\Database_psi_rec_ConvNet_ITER_TensorFlow_v4.mat"
        )
        ind_case = 50000
        mat = mat73.loadmat(filename)
    except:  # Macbook Pro
        folder = os.path.join(
            r"/Users/matte/Dropbox/PhD/RESEARCH_ACTIVITY/ML/Equilibrium_ML/database_equil_ITER_FRIDA"
        )
        ### load probes data and fluxes
        filename = os.path.join(
            folder + r"/Database_psi_rec_ConvNet_ITER_TensorFlow_v4_15000.mat"
        )
        ind_case = 82278
        filename = os.path.join(
            folder + r"/Database_psi_rec_ConvNet_ITER_TensorFlow_v4.mat"
        )
        mat = mat73.loadmat(filename)


for key, val in mat.items():
    print(key)
    exec(key + "=val")

"""
DB_Ax_RZ_flux_test_ConvNet
DB_Ax_flux_test_ConvNet
DB_Betaptest_ConvNet
DB_Iplatot_ConvNet
DB_Jpla_pixel_test_ConvNet
DB_XP_RZ_flux_test_ConvNet
DB_XP_flux_test_ConvNet
DB_coils_curr_test_ConvNet
DB_dpn_test_ConvNet
DB_f_test_ConvNet
DB_fdf_test_ConvNet
DB_li_test_ConvNet
DB_meas_Bpickup_act_test_ConvNet
DB_meas_Bpickup_pla_test_ConvNet
DB_meas_Bpickup_test_ConvNet
DB_meas_fake_fluxloops_act_test_ConvNet
DB_meas_fake_fluxloops_pla_test_ConvNet
DB_meas_fake_fluxloops_test_ConvNet
DB_p_test_ConvNet
DB_psi_pixel_test_ConvNet
DB_psi_pla_pixel_test_ConvNet
DB_psibar_test_ConvNet
DB_res_LHS_pixel_test_ConvNet
DB_res_RHS_pixel_test_ConvNet
DB_separatrix_100_test_ConvNet
DB_separatrix_10_test_ConvNet
DB_separatrix_200_test_ConvNet
DB_separatrix_20_test_ConvNet
GG_Bpickup_Jpla
GG_Bpickup_coil
GG_Psi_J_pixels_XP
GG_Psi_coil_fake_fluxloops_XP
GG_Psi_coil_pixels_XP
GG_Psi_thick_J_fake_fluxloops_XP
RR_pixels_XP
XP_YN
ZZ_pixels_XP
limiter_geo
meshData_pixel_XP
pts_Bpickup
pts_fake_fluxloops_XP
"""

DTYPE = "float32"

DB_meas_Bpickup_test_ConvNet = mat["DB_meas_Bpickup_test_ConvNet"]
DB_psi_pixel_test_ConvNet = mat["DB_psi_pixel_test_ConvNet"]
RR_pixels = mat["RR_pixels_XP"]
ZZ_pixels = mat["ZZ_pixels_XP"]
limiter_geo = mat["limiter_geo"]
DB_separatrix_200_test_ConvNet = mat["DB_separatrix_200_test_ConvNet"]
DB_separatrix_100_test_ConvNet = mat["DB_separatrix_100_test_ConvNet"]
DB_f_test_ConvNet = mat["DB_f_test_ConvNet"]
DB_p_test_ConvNet = mat["DB_p_test_ConvNet"]

XP_YN = mat["XP_YN"]
DB_separatrix_100_test_ConvNet = mat["DB_separatrix_100_test_ConvNet"]
DB_XP_flux_test_ConvNet = mat["DB_XP_flux_test_ConvNet"]
limiter_geo = mat["limiter_geo"]


indices = np.arange(DB_meas_Bpickup_test_ConvNet.shape[0])
id_train, id_test = train_test_split(indices, test_size=0.2, random_state=42)

R0 = 6.1


r_boundary_XP = np.transpose(
    np.squeeze(DB_separatrix_100_test_ConvNet[XP_YN == 1, :, 0])
)
z_boundary_XP = np.transpose(
    np.squeeze(DB_separatrix_100_test_ConvNet[XP_YN == 1, :, 1])
)
r_boundary_lim = np.transpose(
    np.squeeze(DB_separatrix_100_test_ConvNet[XP_YN == 0, :, 0])
)
z_boundary_lim = np.transpose(
    np.squeeze(DB_separatrix_100_test_ConvNet[XP_YN == 0, :, 1])
)

ind_max = 5000

if 0:
    for i in range(20):
        print(i)
        ind_plot_XP = np.random.randint(0, z_boundary_XP.shape[1], ind_max)
        ind_plot_lim = np.random.randint(0, r_boundary_lim.shape[1], ind_max)
        # sns.set_style("ticks", {"axes.facecolor": ".95"})

        fig, ax = plt.subplots(1, 2)
        ax[0].plot(r_boundary_XP[:, ind_plot_XP], z_boundary_XP[:, ind_plot_XP])
        ax[0].plot(limiter_geo[:, 0], limiter_geo[:, 1], "k")
        ax[0].axis("equal")
        ax[0].set_xlabel("r [m]")
        ax[0].set_ylabel("z [m]")
        ax[0].set_title("Diverted equilibria")
        ax[1].plot(r_boundary_lim[:, ind_plot_lim], z_boundary_lim[:, ind_plot_lim])
        ax[1].plot(limiter_geo[:, 0], limiter_geo[:, 1], "k")
        ax[1].axis("equal")
        ax[1].set_xlabel("r [m]")
        ax[1].set_title("Limiter equilibria")
        # plt.show()
        plt.savefig("Equilibria_esampleqw", dpi=300)

for i in plt.get_fignums():
    plt.figure(i)
    plt.savefig("figure%d.png" % i, dpi=300)


# %% ---------------------------------------------
folder = os.path.join(r"/Users/matte/Dropbox/PhD/RESEARCH_ACTIVITY/ML/Equilibrium_ML")

qq = scipy.io.loadmat(os.path.join(folder + r"/coils_data_geo_ref.mat"))
r = qq["coils_data_geo_ref"][0][0][0]
z = qq["coils_data_geo_ref"][0][0][1]
dr = qq["coils_data_geo_ref"][0][0][2]
dz = qq["coils_data_geo_ref"][0][0][3]

coils_geo_r = np.zeros((12, 4))
coils_geo_z = np.zeros((12, 4))

for i in range(12):
    coils_geo_r[i, :] = np.array(
        [r[i] - 0.5 * dr[i], r[i] + 0.5 * dr[i], r[i] + 0.5 * dr[i], r[i] - 0.5 * dr[i]]
    ).ravel()

    coils_geo_z[i, :] = np.array(
        [z[i] - 0.5 * dz[i], z[i] - 0.5 * dz[i], z[i] + 0.5 * dz[i], z[i] + 0.5 * dz[i]]
    ).ravel()

ind_plot = np.random.randint(0, z_boundary_XP.shape[1], 1)[0]
ind_plot = 31800

# sns.set_style("darkgrid")

font = {"size": 13}
import matplotlib

matplotlib.rc("font", **font)

plt.figure()
plt.fill(limiter_geo[:, 0], limiter_geo[:, 1], "C00", label="$\Omega$")
plt.plot(limiter_geo[:, 0], limiter_geo[:, 1], "r", label="$\partial\Omega$")
plt.fill(
    r_boundary_XP[:, ind_plot], z_boundary_XP[:, ind_plot], "C01", label="$\Omega_p$"
)
plt.plot(
    r_boundary_XP[:, ind_plot], z_boundary_XP[:, ind_plot], "k", label="$\Gamma_p$"
)
plt.fill(
    np.transpose(coils_geo_r[0, :]),
    np.transpose(coils_geo_z[0, :]),
    "gray",
    label="coils",
)
for i in range(2, 12):
    plt.fill(np.transpose(coils_geo_r[:, :]), np.transpose(coils_geo_z[:, :]), "gray")
# plt.axis('equal')
# plt.axis('off')
plt.xlabel("r [m]")
plt.ylabel("z [m]")
# plt.xlim(left=0,right=16)
plt.axis("equal")
# plt.gca().set_xlim(0, 10)
plt.legend()
# plt.show()
plt.savefig("ITERlike_geo", dpi=300)


"""
###############################################################################
Routine for various staff
"""


def fun_contourf_sol(
    z,
    RR,
    ZZ,
    sep=np.empty(
        (
            1,
            2,
        )
    ),
    titlefig="",
    axis=None,
):
    plt.figure()
    plt.contourf(RR, ZZ, z, 30)
    plt.axis("equal")
    plt.colorbar()
    plt.plot(limiter_geo[:, 0], limiter_geo[:, 1], "w")
    plt.plot(sep[:, 0], sep[:, 1], "r")
    plt.xlim([RR.min(), RR.max()])
    plt.ylim([ZZ.min(), ZZ.max()])
    plt.xlabel("r")
    plt.ylabel("z")
    plt.title(str(titlefig))
    if axis is not None:
        plt.axis(axis)
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.show()
    return plt


def fun_contour_sol(
    z,
    RR,
    ZZ,
    sep=np.empty(
        (
            1,
            2,
        )
    ),
    titlefig="",
):
    plt.figure()
    plt.contour(RR, ZZ, z, 30)
    plt.axis("equal")
    plt.colorbar()
    plt.plot(limiter_geo[:, 0], limiter_geo[:, 1], "k")
    plt.plot(sep[:, 0], sep[:, 1], "r")
    plt.xlim([RR.min(), RR.max()])
    plt.ylim([ZZ.min(), ZZ.max()])
    plt.xlabel("r")
    plt.ylabel("z")
    plt.title(str(titlefig))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.show()
    return plt


def fun_contour_compare_sol(
    z_ref,
    z,
    RR,
    ZZ,
    sep=np.empty(
        (
            1,
            2,
        )
    ),
    titlefig="",
    ncont=10,
):
    l1 = mlines.Line2D([], [], label="PlaNet")
    l2 = mlines.Line2D([], [], color="black", label="ref")

    plt.figure()
    plt.contour(RR, ZZ, z, ncont)
    plt.colorbar()
    plt.contour(RR, ZZ, z_ref, ncont, colors="black", linestyles="dashed")
    plt.legend(handles=[l1, l2])
    plt.axis("equal")
    plt.plot(limiter_geo[:, 0], limiter_geo[:, 1], "k")
    plt.plot(sep[:, 0], sep[:, 1], "r")
    plt.xlim([RR.min(), RR.max()])
    plt.ylim([ZZ.min(), ZZ.max()])
    plt.xlabel("r")
    plt.ylabel("z")
    plt.title(str(titlefig))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.show()
    return plt


def fun_inpolygon(points, curve):
    points_R, points_Z = points

    # square with legs length 1 and bottom left corner at the origin
    p = path.Path(curve)
    if len(points_R.shape) == 2:
        mask = np.reshape(
            p.contains_points(np.column_stack((points_R.ravel(), points_Z.ravel()))),
            (points_R.shape[0], points_Z.shape[1]),
        )

    else:
        mask = p.contains_points(np.column_stack((points_R.ravel(), points_Z.ravel())))
    return mask


### Scale flux map to [0,1]
class NDMinMaxScaler:
    def fit(self, X):
        self.min_X = np.zeros((X.shape[0], 1))
        self.max_X = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            self.min_X[i] = X[i, :, :].min()
            self.max_X[i] = X[i, :, :].max()
        return self

    def transform(self, X, feature_range=[0, 1]):
        self.a = feature_range[0]
        self.b = feature_range[1]
        X_scaled = np.zeros_like(X)
        for i in range(X.shape[0]):
            if len(X.shape) == 1:
                X_std = (X[i] - self.min_X[i]) / (self.max_X[i] - self.min_X[i])
                X_scaled[i] = X_std * (self.b - self.a) + self.a
            else:
                X_std = (X[i, :, :] - self.min_X[i]) / (self.max_X[i] - self.min_X[i])
                X_scaled[i, :, :] = X_std * (self.b - self.a) + self.a
        return X_scaled


# %% ---------------------------------------------
"""
###############################################################################
Routines for CNN-PINN network
"""
mu0 = 4 * np.pi * 1e-7

hr = RR_pixels[1, 2] - RR_pixels[1, 1]
hz = ZZ_pixels[2, 1] - ZZ_pixels[1, 1]

RR_in = RR_pixels[1:-1, 1:-1]
ZZ_in = ZZ_pixels[1:-1, 1:-1]


### Filters for GS equation
kr = np.array(([0, 0, 0], [1, -2, 1], [0, 0, 0])) * hz**2
kz = np.transpose(np.array(([0, 0, 0], [1, -2, 1], [0, 0, 0]))) * hr**2

alfa = -2 * (hr**2 + hz**2)
Laplace_kernel = np.array(
    ([0, hr**2 / alfa, 0], [hz**2 / alfa, 1, hz**2 / alfa], [0, hr**2 / alfa, 0])
)
Df_dr_kernel = (
    np.array(([0, 0, 0], [+1, 0, -1], [0, 0, 0])) / (2 * hr * alfa) * (hr**2 * hz**2)
)

# Gaussian filter to slightly denoise output
Gaussian_kernel = np.array(([1, 2, 1], [2, 4, 2], [1, 2, 1]), dtype=DTYPE) / (16)
Gaussian_kernel = np.array(
    (
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1],
    ),
    dtype=DTYPE,
) / (273)


# convert everything to tensors
RR_conv_tensor = tf.convert_to_tensor(RR_pixels, dtype=(DTYPE))
RR_in_conv_tensor = tf.convert_to_tensor(RR_in, dtype=(DTYPE))

RR_conv_tensor = tf.expand_dims(tf.expand_dims(RR_conv_tensor, axis=0), axis=-1)
RR_in_conv_tensor = tf.expand_dims(tf.expand_dims(RR_in_conv_tensor, axis=0), axis=-1)

Laplace_kernel = tf.convert_to_tensor(Laplace_kernel, dtype=(DTYPE))
Df_dr_kernel = tf.convert_to_tensor(Df_dr_kernel, dtype=(DTYPE))
Laplace_kernel_tensor = tf.expand_dims(
    tf.expand_dims(Laplace_kernel[::-1, ::-1], axis=-1), axis=-1
)
Df_dr_kernel_tensor = tf.expand_dims(
    tf.expand_dims(Df_dr_kernel[::-1, ::-1], axis=-1), axis=-1
)
Gauss_tensor = tf.expand_dims(
    tf.expand_dims(Gaussian_kernel[::-1, ::-1], axis=-1), axis=-1
)


# routines to compute GS equation via convolution
def fun_GSoperator_NN_conv_batch(f):
    # Compute GS operator via convolution
    Lpsi = tf.nn.conv2d(f, Laplace_kernel_tensor, strides=[1, 1, 1, 1], padding="VALID")
    Dpsi_dr = tf.nn.conv2d(
        f, Df_dr_kernel_tensor, strides=[1, 1, 1, 1], padding="VALID"
    )
    Dpsi_dr = tf.math.divide(Dpsi_dr, RR_in_conv_tensor)

    GS_ope = Lpsi[:, :, :, 0] - Dpsi_dr[:, :, :, 0]
    GS_ope = GS_ope * alfa / (hr**2 * hz**2)

    return GS_ope


def fun_GSoperator_NN_conv_smooth_batch(f):
    # Compute GS operator via convolution
    Lpsi = tf.nn.conv2d(f, Laplace_kernel_tensor, strides=[1, 1, 1, 1], padding="VALID")
    Dpsi_dr = tf.nn.conv2d(
        f, Df_dr_kernel_tensor, strides=[1, 1, 1, 1], padding="VALID"
    )
    Dpsi_dr = tf.math.divide(Dpsi_dr, RR_in_conv_tensor)

    GS_ope = Lpsi - Dpsi_dr
    GS_ope = GS_ope * alfa / (hr**2 * hz**2)

    GS_ope = tf.nn.conv2d(GS_ope, Gauss_tensor, strides=[1, 1, 1, 1], padding="SAME")
    GS_ope = tf.squeeze(GS_ope, axis=-1)

    return GS_ope


def fun_GSoperator_conv_batch(f):
    # Compute GS operator via convolution
    Lpsi = tf.nn.conv2d(f, Laplace_kernel_tensor, strides=[1, 1, 1, 1], padding="VALID")
    Dpsi_dr = tf.nn.conv2d(
        f, Df_dr_kernel_tensor, strides=[1, 1, 1, 1], padding="VALID"
    )
    Dpsi_dr = tf.math.divide(Dpsi_dr, RR_in_conv_tensor)

    GS_ope = Lpsi[:, :, :, 0] - Dpsi_dr[:, :, :, 0]
    GS_ope = GS_ope * alfa / (hr**2 * hz**2)
    jphi = -GS_ope / (mu0 * RR_in)

    return GS_ope, jphi


# %% ---------------------------------------------
"""
###############################################################################
PlaNet equil
"""

DB_psi_pixel_test_ConvNet = mat["DB_psi_pixel_test_ConvNet"]
DB_res_RHS_pixel_test_ConvNet = mat["DB_res_RHS_pixel_test_ConvNet"]
DB_Jpla_pixel_test_ConvNet = mat["DB_Jpla_pixel_test_ConvNet"]
DB_coils_curr_test_ConvNet = mat["DB_coils_curr_test_ConvNet"]
XP_YN = mat["XP_YN"]
DB_f_test_ConvNet = mat["DB_f_test_ConvNet"]
DB_p_test_ConvNet = mat["DB_p_test_ConvNet"]


X_data_load = np.column_stack(
    (
        DB_meas_Bpickup_test_ConvNet,
        DB_coils_curr_test_ConvNet,
        #    DB_f_test_ConvNet,
        DB_p_test_ConvNet,
    )
)

y_data_load = DB_psi_pixel_test_ConvNet
res_RHS_pixel_data_load = DB_res_RHS_pixel_test_ConvNet

n_dim = y_data_load.shape[1]

X_data = X_data_load
y_data = y_data_load


### selec a portion of all the available equilibria
ind_subsample = np.arange(0, X_data.shape[0])
ind_probes = np.arange(0, X_data.shape[1])
X = X_data[ind_subsample, :]
y = y_data[ind_subsample, :, :]
res_RHS_pixel = res_RHS_pixel_data_load[ind_subsample, :, :]
res_RHS_D = DB_res_RHS_pixel_test_ConvNet[ind_subsample, :]

### Standardize input data
scaler = StandardScaler()
X = scaler.fit_transform(X)
np.mean(X[:, 0])


### Train-test split
X_train, X_test = X[id_train, :], X[id_test, :]
y_train, y_test = y[id_train, :, :], y[id_test, :, :]
RHS_train, RHS_test = res_RHS_pixel[id_train, :, :], res_RHS_pixel[id_test, :, :]

Separatrix_train = DB_separatrix_100_test_ConvNet[id_train, :, :]
Separatrix_test = DB_separatrix_100_test_ConvNet[id_test, :, :]


### Print some info
print("N sample train ->", X_train.shape[0])
print("N sample test  ->", X_test.shape[0])

n_output = y_train.shape[1]
n_input = X_train.shape[1]

print("data_X_train.shape =", X_train.shape)
print("data_y_train.shape =", y_train.shape)


### Convert input to tensor
X_train = tf.convert_to_tensor(X_train, dtype=(DTYPE))
y_train = tf.convert_to_tensor(y_train, dtype=(DTYPE))
X_test = tf.convert_to_tensor(X_test, dtype=(DTYPE))
y_test = tf.convert_to_tensor(y_test, dtype=(DTYPE))
RHS_train = tf.convert_to_tensor(RHS_train, dtype=(DTYPE))
RHS_test = tf.convert_to_tensor(RHS_test, dtype=(DTYPE))


### Load pre-trained network
folder = os.path.join(
    r"/Users/matte/Dropbox/PhD/RESEARCH_ACTIVITY/ML/Equilibrium_ML/trained_models"
)

mat_history_j = scipy.io.loadmat(
    os.path.join(folder + r"/history_PlaNet_Equil_kin_65588sample")
)
mat_history_mag = scipy.io.loadmat(
    os.path.join(folder + r"/history_PlaNet_Equil_65588sample_5283epochs.h5")
)

folder = os.path.join(
    r"/Users/matte/Dropbox/PhD/RESEARCH_ACTIVITY/ML/Equilibrium_ML/Physics_Informed_AI_plasma_equilibrium/Notebooks"
)
mat_history_p = scipy.io.loadmat(
    os.path.join(folder + r"/history_PlaNet_Equil_kin_65588_2")
)

mat_history_j = mat_history_j["history"].ravel()
mat_history_p = mat_history_p["history"].ravel()
history_mag = mat_history_mag["history"].ravel()

plt.figure()
plt.plot(np.arange(0, len(history_mag)), history_mag, label="mag only")
plt.plot(
    np.arange(0, len(mat_history_p)), mat_history_p, label="mag + pressure + current"
)
plt.plot(
    np.arange(0, len(mat_history_j)), mat_history_j, label="mag + pressure + current"
)
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("PlaNet equilibrium training history")
plt.yscale("log")
plt.show()
# plt.savefig('training_history_PlaNet_euqil_kin_vs_mag', bbox_inches='tight', dpi=300)


folder = r"./trained_models/"

mat_history_kin = scipy.io.loadmat(
    os.path.join(folder + r"/history_PlaNet_Equil_kin_65588sample")
)["history"]
mat_history_mag = scipy.io.loadmat(
    os.path.join(folder + r"/history_PlaNet_Equil_65588sample_5283epochs.h5")
)["history"]

mat_history_kin_p = scipy.io.loadmat(
    r"./Physics_Informed_AI_plasma_equilibrium/notebooks/history_PlaNet_Equil_kin_65588sample"
)["history"]
mat_history_kin_fp = scipy.io.loadmat(
    r"./Physics_Informed_AI_plasma_equilibrium/notebooks/history_PlaNet_Equil_kin_p_65588sample_1000epochs.h5"
)["history"]

mat_history_kin_p_2 = scipy.io.loadmat(
    r"./Physics_Informed_AI_plasma_equilibrium/notebooks/./history_PlaNet_Equil_kin_65588_2"
)["history"]

mat_history_kin_p = np.column_stack((mat_history_kin_p, mat_history_kin_p_2[:, 1:]))

plt.figure()
plt.plot(mat_history_kin.ravel(), label="mag")
plt.plot(mat_history_mag.ravel(), label="mag + j")
plt.plot(mat_history_kin_p.ravel(), label="mag + p")
# plt.plot(mat_history_kin_fp.ravel(), label='new kin')
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("PlaNet equilibrium training history")
plt.yscale("log")
# plt.savefig('training_history_PlaNet_euqil_kin_vs_mag_2.png', bbox_inches='tight', dpi=300)

plt.show()


df = pd.DataFrame(
    data={
        "epochs": np.arange(mat_history_kin.shape[1]),
        "loss": mat_history_kin.ravel(),
    }
)
df["epochs_2"] = df["epochs"] // 10


def plot_loss(loss, epoch_0=0, agg_factor=10, label=""):
    df = pd.DataFrame(
        data={"epochs_": np.arange(loss.shape[1]) + epoch_0, "loss": loss.ravel()}
    )
    df["epochs"] = df["epochs_"] // agg_factor
    df["epochs"] = df["epochs"] * agg_factor
    # return sns.relplot(
    #     data=df, kind="line",
    #     x="epochs", y="loss", errorbar="sd")
    return sns.lineplot(data=df, y="loss", x="epochs", label=label)


plt.figure()
sns.set_style("darkgrid")
plot_loss(mat_history_mag, label="mag only")
plot_loss(mat_history_kin, label="mag + J profile")
plot_loss(mat_history_kin_p, label="mag + p profile")
plt.yscale("log")
plt.title("PlaNet equilibrium training history")
plt.show()


sns.lineplot(data=df, y="loss", x="epochs_2")
plt.yscale("log")


plt.legend()
plt.show()


# %%


# def conv2D_Norm_activation(x,filters,kernel_size):
#     x = layers.Conv2D(filters=filters ,
#                       kernel_size=kernel_size,
#                       strides=1,
#                       padding='same',
#                       dtype = DTYPE)(x)
#     x = layers.BatchNormalization()(x)
#     # x = tf.keras.activations.relu(x)
#     x = tf.keras.activations.gelu(x)
#     return x


# def eq_model_v3():
#     input_shape = n_input
#     neuron_FC = 2048
#     n_w = 8
#     n_h = 8
#     n_c = int(neuron_FC/(n_h*n_w))
#     interpolation = 'nearest'
#     interpolation = 'bilinear'

#     input_var = tf.keras.Input(input_shape,)
#     x = input_var

#     x = layers.Dense(128,
#                      activation=tf.keras.activations.get('swish'),
#                      kernel_initializer='he_normal',
#                     #  kernel_regularizer=tf.keras.regularizers.L2(0.005),
#                      dtype = DTYPE)(x)

#     x = layers.Dense(128,
#                      activation=tf.keras.activations.get('swish'),
#                      kernel_initializer='he_normal',
#                     #  kernel_regularizer=tf.keras.regularizers.L2(0.005),
#                      dtype = DTYPE)(x)

#     x = layers.Dense(128,
#                      activation=tf.keras.activations.get('swish'),
#                      kernel_initializer='he_normal',
#                     #  kernel_regularizer=tf.keras.regularizers.L2(0.005),
#                      dtype = DTYPE)(x)

#     x = layers.Dense(neuron_FC,
#                      activation=tf.keras.activations.get('swish'),
#                      kernel_initializer='he_normal',
#                     #  kernel_regularizer=tf.keras.regularizers.L2(0.005),
#                      dtype = DTYPE)(x)

#     x = layers.Reshape(target_shape=(n_w, n_h, n_c))(x)


#     x = layers.UpSampling2D(size = (2,2),
#                             interpolation = interpolation)(x)
#     x = conv2D_Norm_activation(x,filters=32,kernel_size=(3, 3))


#     x = layers.UpSampling2D(size = (2,2),
#                             interpolation = interpolation)(x)
#     x = conv2D_Norm_activation(x,filters=16,kernel_size=(3, 3))


#     x = layers.UpSampling2D(size = (2,2),
#                             interpolation = interpolation)(x)
#     x = conv2D_Norm_activation(x,filters=8,kernel_size=(3, 3))


#     # x = layers.UpSampling2D(size = (1,1),
#     #                         interpolation = 'bicubic')(x)
#     # x = conv2D_Norm_activation(x,filters=4,kernel_size=(3, 3))

#     x = layers.Conv2D(filters=1,
#                       kernel_size=(1, 1),
#                       strides=1,
#                       padding='same',
#                       activation='linear',
#                       kernel_initializer='he_normal',
#                       bias_initializer='zeros',
#                       dtype = DTYPE)(x)

#     # x = layers.Resizing(height = y_train.shape[1],width = y_train.shape[2],dtype = DTYPE)(x)
#     outputs = x

#     model = tf.keras.Model(inputs=input_var, outputs=outputs,)
#     model.compile(optimizer='adam',
#                   loss='mse',
#                   run_eagerly=False)

#     return model

"""
###############################################################################
### Physics-Informed Neural Operator
"""


def conv2D_Norm_activation(x, filters, kernel_size, activation="gelu"):
    x = layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=1, padding="same", dtype=DTYPE
    )(x)
    x = layers.BatchNormalization()(x)
    # x = tf.keras.activations.relu(x)
    x = (
        tf.keras.activations.tanh(x)
        if activation == "tanh"
        else tf.keras.activations.gelu(x)
    )
    return x


def PlaNet_Equil_Neural_Opt():
    input_shape_fun = n_input
    input_query_r = 64
    input_query_z = 64
    neuron_FC = 2048
    n_w = 8
    n_h = 8
    n_c = int(neuron_FC / (n_h * n_w))
    interpolation = "nearest"
    interpolation = "bilinear"

    input_fun = tf.keras.Input(
        shape=(input_shape_fun,), name="function"
    )  # meas + active currents (+ profiles)
    input_query_RR = tf.keras.Input(
        shape=(
            input_query_r,
            input_query_z,
            1,
        ),
        name="R_grid_query",
    )  # input coordinates (query pts)
    input_query_ZZ = tf.keras.Input(
        shape=(
            input_query_r,
            input_query_z,
            1,
        ),
        name="Z_grid_query",
    )  # input coordinates (query pts)

    inputs = [input_fun, input_query_RR, input_query_ZZ]

    # Branch net
    x = layers.Dense(
        128,
        activation=tf.keras.activations.get("swish"),
        kernel_initializer="he_normal",
        #  kernel_regularizer=tf.keras.regularizers.L2(0.005),
        dtype=DTYPE,
    )(input_fun)

    x = layers.Dense(
        128,
        activation=tf.keras.activations.get("swish"),
        kernel_initializer="he_normal",
        #  kernel_regularizer=tf.keras.regularizers.L2(0.005),
        dtype=DTYPE,
    )(x)

    out_branch = layers.Dense(
        128,
        activation=tf.keras.activations.get("swish"),
        kernel_initializer="he_normal",
        #  kernel_regularizer=tf.keras.regularizers.L2(0.005),
        dtype=DTYPE,
    )(x)

    # Trunk net
    x_r = input_query_RR
    for i in range(1):
        x_r = input_query_RR if i == 0 else x_r
        x_r = conv2D_Norm_activation(x_r, filters=i + 1, kernel_size=(3, 3))
        x_r = layers.MaxPooling2D(pool_size=(2, 2))(x_r)

    x_z = input_query_ZZ
    for i in range(1):
        x_z = input_query_ZZ if i == 0 else x_z
        x_z = conv2D_Norm_activation(x_z, filters=i + 1, kernel_size=(3, 3))
        x_z = layers.MaxPooling2D(pool_size=(2, 2))(x_z)

    out_trunk = layers.Concatenate()([x_r, x_z])
    out_trunk = layers.Flatten()(out_trunk)
    out_trunk = layers.Dense(
        128,
        activation=tf.keras.activations.get("gelu"),
        kernel_initializer="he_normal",
        dtype=DTYPE,
    )(out_trunk)

    for i in range(2):
        out_trunk = layers.Dense(
            128,
            activation=tf.keras.activations.get("gelu"),
            kernel_initializer="he_normal",
            dtype=DTYPE,
        )(out_trunk)

    # Multiply layer
    out_multiply = layers.Multiply(name="Multiply")([out_branch, out_trunk])

    # conv2d-based decoder
    x_dec = layers.Dense(
        neuron_FC,
        activation=tf.keras.activations.get("gelu"),
        kernel_initializer="he_normal",
        #  kernel_regularizer=tf.keras.regularizers.L2(0.005),
        dtype=DTYPE,
    )(out_multiply)

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

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
    )
    model.compile(optimizer="adam", loss="mse", run_eagerly=False)

    return model


model = PlaNet_Equil_Neural_Opt()
model.summary()
model.save("PlaNet_Equil_Neural_Opt.keras")


# %% ---------------------------------------------

scaler_r = StandardScaler()
scaler_z = StandardScaler()
scaler_r.fit(RR_pixels.reshape(-1, 1))
scaler_z.fit(ZZ_pixels.reshape(-1, 1))
RR_pixels_scaled = scaler_r.transform(RR_pixels.reshape(-1, 1)).reshape(RR_pixels.shape)
ZZ_pixels_scaled = scaler_z.transform(ZZ_pixels.reshape(-1, 1)).reshape(RR_pixels.shape)

RR_pixels_scaled = np.tile(RR_pixels_scaled, (X_train.shape[0], 1, 1))
ZZ_pixels_scaled = np.tile(ZZ_pixels_scaled, (X_train.shape[0], 1, 1))

RR_pixels_scaled = tf.convert_to_tensor(RR_pixels_scaled, dtype=(DTYPE))
ZZ_pixels_scaled = tf.convert_to_tensor(ZZ_pixels_scaled, dtype=(DTYPE))

RR_pixels_scaled = tf.expand_dims(RR_pixels_scaled, axis=-1)
ZZ_pixels_scaled = tf.expand_dims(ZZ_pixels_scaled, axis=-1)


# %% ---------------------------------------------

batch_size = 64

train_ds = tf.data.Dataset.from_tensor_slices(
    (
        X_train,
        y_train,
        #  res_RHS_pixel_train[:,1:-1,1:-1],
        RR_pixels_scaled,
        ZZ_pixels_scaled,
    )
).batch(batch_size)

test_ds = tf.data.Dataset.from_tensor_slices(
    (
        X_test,
        y_test,
        #  res_RHS_pixel_train[:,1:-1,1:-1],
        RR_pixels_scaled[: X_test.shape[0], :, :, :],
        ZZ_pixels_scaled[: X_test.shape[0], :, :, :],
    )
).batch(batch_size)

# %% ---------------------------------------------


model_filename = os.path.join(
    r"./Physics_Informed_AI_plasma_equilibrium/model_PlaNet_NeuralOp_65588_3500epochs.h5"
)
PlaNet_NeuralOpt = keras.models.load_model(model_filename, compile=False)

# from model_tf import eq_model_v3
# PlaNet_equil_kin = eq_model_v3()
# PlaNet_equil_kin.load_weights(model_filename)
# train_and_checkpoint(model, manager)


###

# Train
separatrix_ref = Separatrix_train
psi_NN_kin = PlaNet_NeuralOpt.predict([X_train, RR_pixels_scaled, ZZ_pixels_scaled])
psi_ref = y_train.numpy()
GS_ope_ref = RHS_train[:, 1:-1, 1:-1].numpy()

GSope_NN_kin = fun_GSoperator_NN_conv_smooth_batch(psi_NN_kin).numpy()

vec_psi_kin_train = np.squeeze(psi_NN_kin).ravel()
vec_psi_ref_train = psi_ref.ravel()

vec_GSope_kin_train = GSope_NN_kin.ravel()
vec_GSope_ref_train = GS_ope_ref.ravel()

r2_score_psi_kin_train = r2_score(
    vec_psi_ref_train, vec_psi_kin_train
)  # 0.9999826736615185
r2_score_GSope_kin_train = r2_score(
    vec_GSope_ref_train, vec_GSope_kin_train
)  # 0.9997634814110631

mse_psi_kin_train = mean_squared_error(
    vec_psi_ref_train, vec_psi_kin_train
)  # 0.00085132074
mse_GSope_kin_train = mean_squared_error(
    vec_GSope_ref_train, vec_GSope_kin_train
)  # 0.0037826695

# Test
separatrix_ref = Separatrix_test
psi_NN_kin = PlaNet_NeuralOpt.predict(
    [
        X_test,
        RR_pixels_scaled[: X_test.shape[0], :, :, :],
        ZZ_pixels_scaled[: X_test.shape[0], :, :, :],
    ]
)

psi_ref = y_test.numpy()
GS_ope_ref = RHS_test[:, 1:-1, 1:-1].numpy()

GSope_NN_kin = fun_GSoperator_NN_conv_smooth_batch(psi_NN_kin).numpy()

vec_psi_kin_test = np.squeeze(psi_NN_kin).ravel()
vec_psi_ref_test = psi_ref.ravel()

vec_GSope_kin_test = GSope_NN_kin.ravel()
vec_GSope_ref_test = GS_ope_ref.ravel()

r2_score_psi_kin_test = r2_score(
    vec_psi_ref_test, vec_psi_kin_test
)  # 0.9999825504177072
r2_score_GSope_kin_test = r2_score(
    vec_GSope_ref_test, vec_GSope_kin_test
)  # 0.9997612930908663

mse_psi_kin_test = mean_squared_error(
    vec_psi_ref_test, vec_psi_kin_test
)  # 0.00085321494
mse_GSope_kin_test = mean_squared_error(
    vec_GSope_ref_test, vec_GSope_kin_test
)  # 0.0037938117

s_scatter = 5
n_sup = int(0.03 * vec_GSope_ref_test.shape[0])
# plt.figure()
# plt.scatter(vec_GSope_ref_test[:n_sup],vec_GSope_mag_test[:n_sup], s_scatter, color = 'C2',
#             marker = '.',label = 'val. - $R^2$ = {:4.4f}'.format(r2_score_GSope_mag_test))
# plt.scatter(vec_GSope_ref_train[:n_sup],vec_GSope_mag_train[:n_sup], s_scatter, color = 'C3',
#             marker = '.',label = 'train - $R^2$ = {:4.4f}'.format(r2_score_GSope_mag_train))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.gca().set_xlabel('$\Delta^*\psi$')
# plt.gca().set_title('GS operator (mag only)')
# plt.gca().set_ylabel('$\Delta^*\psi^p$')
# plt.legend()
# plt.savefig('scatterplot_PlaNet_euqil_mag_2.png', bbox_inches='tight', dpi=300)


plt.figure()
plt.scatter(
    vec_GSope_ref_test[:n_sup],
    vec_GSope_kin_test[:n_sup],
    s_scatter,
    color="C2",
    marker=".",
    label="val. - $R^2$ = {:4.4f}".format(r2_score_GSope_kin_test),
)
plt.scatter(
    vec_GSope_ref_train[:n_sup],
    vec_GSope_kin_train[:n_sup],
    s_scatter,
    color="C3",
    marker=".",
    label="train - $R^2$ = {:4.4f}".format(r2_score_GSope_kin_train),
)
plt.gca().set_aspect("equal", adjustable="box")
plt.legend()
plt.gca().set_xlabel("$\Delta^*\psi$")
plt.gca().set_ylabel("$\Delta^*\psi^p$")
plt.gca().set_title("GS operator (mag + j)")
plt.gca().legend()
plt.savefig("scatterplot_PlaNet_euqil_j_2.png", bbox_inches="tight", dpi=300)


# %% ---------------------------------------------
#

ind_test = int(np.random.randint(0, X_test.shape[0], 1))
psi_ref = y_test[ind_test, :, :].numpy()
X_test_i = X_test[ind_test : ind_test + 1, :]

fun_contourf_sol(psi_ref, RR_pixels, ZZ_pixels)

x = np.linspace(5, 7, 64)
y = np.linspace(-3.5, -1.5, 64)
xv_0, yv_0 = np.meshgrid(x, y, indexing="xy")

xv = scaler_r.transform(xv_0.reshape(-1, 1)).reshape(xv_0.shape)
yv = scaler_z.transform(yv_0.reshape(-1, 1)).reshape(xv_0.shape)

xv = tf.convert_to_tensor(xv, dtype=(DTYPE))
yv = tf.convert_to_tensor(yv, dtype=(DTYPE))

xv = tf.expand_dims(tf.expand_dims(xv, axis=0), axis=-1)
yv = tf.expand_dims(tf.expand_dims(yv, axis=0), axis=-1)

predictions = PlaNet_NeuralOpt.predict([X_test_i, xv, yv])[0, :, :, 0]

plt.figure()
plt.scatter(RR_pixels_scaled[0, :, :, 0], ZZ_pixels_scaled[0, :, :, 0])
plt.scatter(xv[0, :, :, 0], yv[0, :, :, 0])

fun_contourf_sol(predictions, xv_0, yv_0)


# from scipy.interpolate import RegularGridInterpolator
# interp_func = RegularGridInterpolator((RR_pixels[0,:], ZZ_pixels[:,0]), psi_ref)
# result = interp_func(np.column_stack([
#     xv_0.reshape(-1,1),
#     yv_0.reshape(-1,1),
# ])).reshape(RR_pixels.shape)

axis = [4, 8, -5, -2]
plt = fun_contourf_sol(psi_ref, RR_pixels, ZZ_pixels, axis=axis)


# %% ---------------------------------------------
# GS_ope_ref = fun_GSoperator_NN_conv_smooth_batch(tf.expand_dims(y_test, axis=-1)).numpy()

ind_plot = 316
ind_plot = 7666
ind_plot = 10826

printfig = False
if printfig == True:
    for i in range(5):
        ind_plot = np.random.randint(0, psi_NN_kin.shape[0], 1)[0]
        # ind_plot = 0
        separatrix_ref_ii = separatrix_ref[ind_plot, :, :]

        psi_NN_kin_ii = np.squeeze(psi_NN_kin[ind_plot, :, :, 0])
        psi_NN_mag_ii = np.squeeze(psi_NN_mag[ind_plot, :, :, 0])
        psi_ref_ii = np.squeeze(psi_ref[ind_plot, :, :])

        GSope_NN_kin_ii = np.squeeze(GSope_NN_kin[ind_plot, :, :])
        GSope_NN_mag_ii = np.squeeze(GSope_NN_mag[ind_plot, :, :])
        GS_ope_ref_ii = np.squeeze(GS_ope_ref[ind_plot, :, :])

        fun_contour_compare_sol(
            psi_NN_kin_ii,
            psi_ref_ii,
            RR_pixels,
            ZZ_pixels,
            separatrix_ref_ii,
            "Poloidal flux (mag + profiles)",
            20,
        )
        if printfig:
            plt.savefig(
                "Psi_kin_case{:d}.png".format(ind_plot), bbox_inches="tight", dpi=300
            )

        fun_contour_compare_sol(
            psi_NN_mag_ii,
            psi_ref_ii,
            RR_pixels,
            ZZ_pixels,
            separatrix_ref_ii,
            "Poloidal flux (mag only)",
            20,
        )
        if printfig:
            plt.savefig(
                "Psi_mag_case{:d}.png".format(ind_plot), bbox_inches="tight", dpi=300
            )

        ind_NaN_in = ~fun_inpolygon((RR_in, ZZ_in), separatrix_ref_ii)
        GSope_NN_kin_ii[ind_NaN_in] = 0
        GSope_NN_mag_ii[ind_NaN_in] = 0
        GS_ope_ref_ii[ind_NaN_in] = 0

        printfig = True
        plt = fun_contour_compare_sol(
            GS_ope_ref_ii,
            GSope_NN_kin_ii,
            RR_in,
            ZZ_in,
            separatrix_ref_ii,
            "GS op. (mag + profiles)",
        )
        if printfig:
            plt.savefig(
                "GS_ope_kin_case{:d}.png".format(ind_plot), bbox_inches="tight", dpi=300
            )

        plt = fun_contour_compare_sol(
            GS_ope_ref_ii,
            GSope_NN_mag_ii,
            RR_in,
            ZZ_in,
            separatrix_ref_ii,
            "GS op. (mag only)",
        )
        if printfig:
            plt.savefig(
                "GS_ope_mag_case{:d}.png".format(ind_plot), bbox_inches="tight", dpi=300
            )

        plt = fun_contourf_sol(
            100 * np.abs(psi_NN_kin_ii - psi_ref_ii) / np.max(np.abs(psi_ref_ii)),
            RR_pixels,
            ZZ_pixels,
            separatrix_ref_ii,
            "Poloidal flux error [%] (mag + profiles)",
        )
        if printfig:
            plt.savefig(
                "Psi_kin_error_case{:d}.png".format(ind_plot),
                bbox_inches="tight",
                dpi=300,
            )

        plt = fun_contourf_sol(
            100 * np.abs(psi_NN_mag_ii - psi_ref_ii) / np.max(np.abs(psi_ref_ii)),
            RR_pixels,
            ZZ_pixels,
            separatrix_ref_ii,
            "Poloidal flux error [%] (mag only)",
        )
        if printfig:
            plt.savefig(
                "Psi_mag_error_case{:d}.png".format(ind_plot),
                bbox_inches="tight",
                dpi=300,
            )

        plt = fun_contourf_sol(
            100
            * np.abs(GSope_NN_kin_ii - GS_ope_ref_ii)
            / np.max(np.abs(GS_ope_ref_ii)),
            RR_in,
            ZZ_in,
            separatrix_ref_ii,
            "GS op. [%] (mag + profiles)",
        )
        if printfig:
            plt.savefig(
                "GS_ope_kin_error_case{:d}.png".format(ind_plot),
                bbox_inches="tight",
                dpi=300,
            )

        plt = fun_contourf_sol(
            100
            * np.abs(GSope_NN_mag_ii - GS_ope_ref_ii)
            / np.max(np.abs(GS_ope_ref_ii)),
            RR_in,
            ZZ_in,
            separatrix_ref_ii,
            "GS op. [%] (mag only)",
        )
        if printfig:
            plt.savefig(
                "GS_ope_mag _error_case{:d}.png".format(ind_plot),
                bbox_inches="tight",
                dpi=300,
            )


printfig = False
if printfig == True:
    for i in range(0, 5):
        ind_plot = np.random.randint(0, X_train[0].shape[0], 1)[0]
        psi_NN_kin_ii = np.squeeze(psi_NN_kin[ind_plot, :, :, 0])
        psi_NN_mag_ii = np.squeeze(psi_NN_mag[ind_plot, :, :, 0])
        psi_ref_ii = np.squeeze(psi_ref[ind_plot, :, :])
        separatrix_ref_ii = separatrix_ref[ind_plot, :, :]

        fun_contour_compare_sol(
            psi_ref_ii,
            psi_NN_kin_ii,
            RR_pixels,
            ZZ_pixels,
            separatrix_ref_ii,
            "Poloidal flux (mag + prof)",
        )

        plt.savefig(
            "example_Eq_rec_kin{:d}.png".format(ind_plot), bbox_inches="tight", dpi=300
        )

        fun_contour_compare_sol(
            psi_ref_ii,
            psi_NN_mag_ii,
            RR_pixels,
            ZZ_pixels,
            separatrix_ref_ii,
            "Poloidal flux (mag only)",
        )
        plt.savefig(
            "example_Eq_rec_mag{:d}.png".format(ind_plot), bbox_inches="tight", dpi=300
        )


"""
###############################################################################
PlaNet XPlim (classification + separatrix flux regression)
"""

### Prepare PlaNet equil output data to be fed to the next network
Psi_train = PlaNet_equil_kin.predict(X_train)
Psi_test = PlaNet_equil_kin.predict(X_test)

Psi_train = np.squeeze(Psi_train)
Psi_test = np.squeeze(Psi_test)


# Psi_b_train = DB_XP_flux_test_ConvNet[id_train]
# Psi_b_test = DB_XP_flux_test_ConvNet[id_test]

# NDscaler = NDMinMaxScaler()
# NDscaler.fit(DB_psi_pixel_test_ConvNet[id_train,:,:])
# Psi_train_scaled = NDscaler.transform(DB_psi_pixel_test_ConvNet[id_train,:,:],feature_range = [0,1])
# Psi_b_train_scaled = NDscaler.transform(Psi_b_train,feature_range = [0,1])

# NDscaler = NDMinMaxScaler()
# NDscaler.fit(DB_psi_pixel_test_ConvNet[id_test,:,:])
# Psi_test_scaled = NDscaler.transform(DB_psi_pixel_test_ConvNet[id_test,:,:],feature_range = [0,1])
# Psi_b_test_scaled = NDscaler.transform(Psi_b_test,feature_range = [0,1])


### Prepare reference data
Psi_b_train = DB_XP_flux_test_ConvNet[id_train]
Psi_b_test = DB_XP_flux_test_ConvNet[id_test]

XP_YN_train = LabelEncoder().fit_transform(XP_YN[id_train])
XP_YN_test = LabelEncoder().fit_transform(XP_YN[id_test])


### Scaling poloidal flux maps and separatrix flux in range [0,1]
NDscaler_train = NDMinMaxScaler()
NDscaler_train.fit(Psi_train)
Psi_train_scaled = NDscaler_train.transform(Psi_train, feature_range=[0, 1])
Psi_b_train_scaled = NDscaler_train.transform(Psi_b_train, feature_range=[0, 1])

NDscaler_test = NDMinMaxScaler()
NDscaler_test.fit(Psi_test)
Psi_test_scaled = NDscaler_test.transform(Psi_test, feature_range=[0, 1])
Psi_b_test_scaled = NDscaler_test.transform(Psi_b_test, feature_range=[0, 1])


### Load model and history and perform predictions
PlaNet_model_XPlim = tf.keras.models.load_model(
    os.path.join(folder + r"/model_PlaNet_XPlim_65588sample.h5")
)
history_XPlim = scipy.io.loadmat(
    os.path.join(folder + r"/history_PlaNet_XPlim_65588sample.h5")
)["history"]
history_XPlim = history_XPlim.ravel()

plt.figure()
plt.plot(np.arange(0, len(history_XPlim)), history_XPlim)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("PlaNet XPlim training history")
plt.yscale("log")
plt.show()


out_clas_proba_train, out_regr_train = PlaNet_model_XPlim.predict(Psi_train_scaled)
out_clas_proba_test, out_regr_test = PlaNet_model_XPlim.predict(Psi_test_scaled)

out_regr_train = out_regr_train.ravel()
out_regr_test = out_regr_test.ravel()


### Check classification outcomes
out_clas_train = np.round(out_clas_proba_train)
out_clas_test = np.round(out_clas_proba_test)

print("f1 train =", f1_score(XP_YN_train, out_clas_train))  # 0.9880188878708859
print(
    "accuracy train =", accuracy_score(XP_YN_train, out_clas_train)
)  # 0.989632249801793
print("f1 test =", f1_score(XP_YN_test, out_clas_test))  # 0.9883639422402916
print(
    "accuracy test =", accuracy_score(XP_YN_test, out_clas_test)
)  # 0.9898768142456397


FPR_train, TPR_train, threshold1 = roc_curve(XP_YN_train, out_clas_proba_train)
FPR_test, TPR_test, threshold1 = roc_curve(XP_YN_test, out_clas_proba_test)

roc_auc_train = roc_auc_score(XP_YN_train, out_clas_proba_train)
roc_auc_test = roc_auc_score(XP_YN_test, out_clas_proba_test)

plt.figure()
plt.title("Equilibrium classification - ROC")
plt.plot(FPR_train, TPR_train, label="train, ROC area = {:5.5f}".format(roc_auc_train))
plt.plot(
    FPR_test, TPR_test, "--", label="val., ROC area = {:5.5f}".format(roc_auc_test)
)
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.legend()
plt.axis("equal")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()
# plt.savefig('classif_ROC.png', bbox_inches='tight', dpi=300)


### Check regression outcomes
R2_regr_Psi_b_train = r2_score(Psi_b_train_scaled, out_regr_train)
R2_regr_Psi_b_test = r2_score(Psi_b_test_scaled, out_regr_test)
print("R2 train =", R2_regr_Psi_b_train)  # 0.9935119536711146
print("R2 test =", R2_regr_Psi_b_test)  # 0.9936548547131024


s_scatter = 5
plt.figure()
plt.scatter(
    Psi_b_train_scaled,
    out_regr_train,
    s_scatter,
    color="C02",
    marker=".",
    label="train - $R^2$ = {:4.4f}".format(R2_regr_Psi_b_train),
)
plt.scatter(
    Psi_b_test_scaled,
    out_regr_test,
    s_scatter,
    color="C03",
    marker=".",
    label="val. - $R^2$ = {:4.4f}".format(R2_regr_Psi_b_test),
)
plt.axis("equal")
plt.xlabel("$\psi_{b,scaled}$")
plt.ylabel("$\hat{\psi}_{b,scaled}$")
plt.title("$\hat{\psi}_{b,scaled}$ reconstruction")
plt.legend()
plt.show()
# plt.savefig('psib_regr_scatterplot.png', bbox_inches='tight', dpi=300)


### Check wrong classifications
ind_check = np.where(out_clas_train.ravel() != XP_YN_train.ravel())[0]

ind_sel = 1

printfig = False
for i in range(10):
    ind_sel = np.random.randint(0, ind_check.shape[0], 1)[0]
    psi_grid = np.squeeze(Psi_train[ind_check[ind_sel], :, :])
    psi_b = Psi_b_train[ind_check[ind_sel]]
    separatrix_ii = Separatrix_train[ind_check[ind_sel], :, :]

    if XP_YN_train[ind_check[ind_sel]] == 1:
        titlefig = "           Diverted, misclassified as limiter"
    else:
        titlefig = "           Limiter, misclassified as diverted"

    plt.figure()
    plt.contourf(RR_pixels, ZZ_pixels, psi_grid, 25)
    plt.colorbar()
    plt.contour(RR_pixels, ZZ_pixels, psi_grid, [psi_b], colors="c")
    # plt.plot(separatrix_ii[:,0],separatrix_ii[:,1],color ='g')
    plt.plot(limiter_geo[:, 0], limiter_geo[:, 1], "w")
    plt.axis("equal")
    plt.title(str(titlefig))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.show()
    if printfig == True:
        plt.savefig(
            "example_misclassification_{:d}.png".format(ind_sel),
            bbox_inches="tight",
            dpi=300,
        )


"""
###############################################################################
PlaNet separatrix
"""

### Prepare inputs (the output of the previous networks)
Psi_b_train = out_regr_train
Psi_b_train_scaled = NDscaler_train.transform(Psi_b_train, feature_range=[0, 1])

Psi_b_test = out_regr_test
Psi_b_test_scaled = NDscaler_test.transform(Psi_b_test, feature_range=[0, 1])


ind_XP_train_ref = np.where(XP_YN[id_train] == 1, True, False)  # diverted equilibria
ind_XP_test_ref = np.where(XP_YN[id_test] == 1, True, False)  # diverted equilibria
ind_lim_train_ref = np.where(XP_YN[id_train] == 1, False, True)  # limiter equilibria
ind_lim_test_ref = np.where(XP_YN[id_test] == 1, False, True)  # limiter equilibria


Psi_train_XP_scaled = Psi_train_scaled[ind_XP_train_ref, :, :]
Psi_test_XP_scaled = Psi_test_scaled[ind_XP_test_ref, :, :]
Psi_train_lim_scaled = Psi_train_scaled[ind_lim_train_ref, :, :]
Psi_test_lim_scaled = Psi_test_scaled[ind_lim_test_ref, :, :]

Psi_b_train_XP_scaled = Psi_b_train_scaled[ind_XP_train_ref]
Psi_b_test_XP_scaled = Psi_b_test_scaled[ind_XP_test_ref]
Psi_b_train_lim_scaled = Psi_b_train_scaled[ind_lim_train_ref]
Psi_b_test_lim_scaled = Psi_b_test_scaled[ind_lim_test_ref]

input_PlaNet_Separatrix_XP_train = [Psi_train_XP_scaled, Psi_b_train_XP_scaled]
input_PlaNet_Separatrix_XP_test = [Psi_test_XP_scaled, Psi_b_test_XP_scaled]


input_PlaNet_Separatrix_lim_train = [Psi_train_lim_scaled, Psi_b_train_lim_scaled]
input_PlaNet_Separatrix_lim_test = [Psi_test_lim_scaled, Psi_b_test_lim_scaled]


### Prepare reference data
Separatrix_ref_train = DB_separatrix_100_test_ConvNet[id_train, 0:-1, :]
Separatrix_ref_test = DB_separatrix_100_test_ConvNet[id_test, 0:-1, :]

Separatrix_ref_train[:, :, 0] = Separatrix_ref_train[:, :, 0] - R0
Separatrix_ref_test[:, :, 0] = Separatrix_ref_test[:, :, 0] - R0


Separatrix_ref_XP_train = Separatrix_ref_train[ind_XP_train_ref, :, :]
Separatrix_ref_XP_test = Separatrix_ref_test[ind_XP_test_ref, :, :]

Separatrix_ref_lim_train = Separatrix_ref_train[ind_lim_train_ref, :, :]
Separatrix_ref_lim_test = Separatrix_ref_test[ind_lim_test_ref, :, :]


fig, ax = plt.subplots(1, 2)
ax[0].plot(Separatrix_ref_XP_train[:, :, 0] + R0, Separatrix_ref_XP_train[:, :, 1])
ax[0].plot(limiter_geo[:, 0], limiter_geo[:, 1])


### Load models and check hostory
PlaNet_separatrix_lim = tf.keras.models.load_model(
    os.path.join(folder + r"/model_PlaNet_boundary_limiter_65588sample.h5")
)
PlaNet_separatrix_XP = tf.keras.models.load_model(
    os.path.join(folder + r"/model_PlaNet_boundary_Xpoint_65588sample.h5")
)

history_separatrix_lim = scipy.io.loadmat(
    os.path.join(folder + r"/history_PlaNet_boundary_limiter_65588sample.h5")
)["history"]
history_separatrix_XP = scipy.io.loadmat(
    os.path.join(folder + r"/history_PlaNet_boundary_Xpoint_65588sample.h5")
)["history"]

history_separatrix_lim = history_separatrix_lim.ravel()
history_separatrix_XP = history_separatrix_XP.ravel()

plt.figure()
plt.plot(
    np.arange(0, len(history_separatrix_lim)), history_separatrix_lim, label="Limiter"
)
plt.plot(np.arange(0, len(history_separatrix_XP)), history_separatrix_XP, label="XP")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.title("PlaNet Separatrix training history")
plt.yscale("log")
plt.show()
# plt.savefig('training_history_PlaNet_separatrix_XP_vs_lim.png', bbox_inches='tight', dpi=300)


### Predictions for XP case
Separatrix_XP_train = PlaNet_separatrix_XP.predict(input_PlaNet_Separatrix_XP_train)
Separatrix_XP_test = PlaNet_separatrix_XP.predict(input_PlaNet_Separatrix_XP_test)


def fun_compare_separatrix(Psi_grid, separatrix_ref, separatrix_NN):
    if len(separatrix_NN.shape) > 2:
        separatrix_NN = np.squeeze(separatrix_NN)
    plt.figure()
    plt.contourf(RR_pixels, ZZ_pixels, Psi_grid, 50)
    # plt.colorbar()
    ind = np.row_stack(
        (
            np.arange(0, separatrix_ref.shape[0]).reshape(-1, 1),
            np.array([[separatrix_ref.shape[0] - 1], [0]]),
        )
    )
    plt.plot(separatrix_ref[ind, 0] + R0, separatrix_ref[ind, 1], color="k")
    plt.plot(separatrix_NN[ind, 0] + R0, separatrix_NN[ind, 1], "--c")
    plt.plot(limiter_geo[:, 0], limiter_geo[:, 1], color="k")
    plt.axis("equal")
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.show()
    return plt


for i in range(0, 5):
    ind_plot = np.random.randint(0, X_train[0].shape[0], 1)[0]
    Psi_grid = np.squeeze(input_PlaNet_Separatrix_XP_train[0][ind_plot, :, :])
    separatrix_ref = Separatrix_ref_XP_train[ind_plot, :, :]
    # input_PlaNet_Separatrix_XP_train_ii = [Psi_train_XP_scaled[ind_plot:ind_plot+1,:,:],
    #                                        Psi_b_train_XP_scaled[ind_plot:ind_plot+1]]
    # separatrix_NN = PlaNet_separatrix_XP.predict(input_PlaNet_Separatrix_XP_train_ii)
    separatrix_NN = Separatrix_XP_train[ind_plot, :, :]
    fun_compare_separatrix(Psi_grid, separatrix_ref, separatrix_NN)
    # plt.savefig('example_separatrix_rec_XP{:d}.png'.format(ind_plot), bbox_inches='tight', dpi=300)


print(
    "R2 train on R coordinate -> ",
    r2_score(Separatrix_ref_XP_train[:, :, 0], Separatrix_XP_train[:, :, 0]),
)
print(
    "R2 test on R coordinate -> ",
    r2_score(Separatrix_ref_XP_test[:, :, 0], Separatrix_XP_test[:, :, 0]),
)
print(
    "R2 train on Z coordinate -> ",
    r2_score(Separatrix_ref_XP_train[:, :, 1], Separatrix_XP_train[:, :, 1]),
)
print(
    "R2 test on Z coordinate -> ",
    r2_score(Separatrix_ref_XP_test[:, :, 1], Separatrix_XP_test[:, :, 1]),
)

print(
    "MSE train on R coordinate -> ",
    mean_squared_error(Separatrix_ref_XP_train[:, :, 0], Separatrix_XP_train[:, :, 0]),
)
print(
    "MSE test on R coordinate -> ",
    mean_squared_error(Separatrix_ref_XP_test[:, :, 0], Separatrix_XP_test[:, :, 0]),
)
print(
    "MSE train on Z coordinate -> ",
    mean_squared_error(Separatrix_ref_XP_train[:, :, 1], Separatrix_XP_train[:, :, 1]),
)
print(
    "MSE test on Z coordinate -> ",
    mean_squared_error(Separatrix_ref_XP_test[:, :, 1], Separatrix_XP_test[:, :, 1]),
)

# ### With reference Psi_b
# R2 train on R coordinate ->  0.9933837231689878
# R2 test on R coordinate ->  0.9933686316681145
# R2 train on Z coordinate ->  0.995404419238913
# R2 test on Z coordinate ->  0.9954547583598966

# ### With Psi_b computed by the PlaNet XPlim net
# R2 train on R coordinate ->  0.9926864886342565
# R2 test on R coordinate ->  0.9927343277026006
# R2 train on Z coordinate ->  0.9952967558545057
# R2 test on Z coordinate ->  0.9953576029917113


### Predictions for limiter case
Separatrix_lim_train = PlaNet_separatrix_lim.predict(input_PlaNet_Separatrix_lim_train)
Separatrix_lim_test = PlaNet_separatrix_lim.predict(input_PlaNet_Separatrix_lim_test)

for i in range(0, 5):
    ind_plot = np.random.randint(0, X_train[0].shape[0], 1)[0]
    Psi_grid = np.squeeze(input_PlaNet_Separatrix_lim_train[0][ind_plot, :, :])
    separatrix_ref = Separatrix_ref_lim_train[ind_plot, :, :]
    # input_PlaNet_Separatrix_XP_train_ii = [Psi_train_XP_scaled[ind_plot:ind_plot+1,:,:],
    #                                        Psi_b_train_XP_scaled[ind_plot:ind_plot+1]]
    # separatrix_NN = PlaNet_separatrix_XP.predict(input_PlaNet_Separatrix_XP_train_ii)
    separatrix_NN = Separatrix_lim_train[ind_plot, :, :]
    fun_compare_separatrix(Psi_grid, separatrix_ref, separatrix_NN)
    # plt.savefig('example_separatrix_rec_lim{:d}.png'.format(ind_plot), bbox_inches='tight', dpi=300)


print(
    "R2 train on R coordinate -> ",
    r2_score(Separatrix_ref_lim_train[:, :, 0], Separatrix_lim_train[:, :, 0]),
)
print(
    "R2 test on R coordinate -> ",
    r2_score(Separatrix_ref_lim_test[:, :, 0], Separatrix_lim_test[:, :, 0]),
)
print(
    "R2 train on Z coordinate -> ",
    r2_score(Separatrix_ref_lim_train[:, :, 1], Separatrix_lim_train[:, :, 1]),
)
print(
    "R2 test on Z coordinate -> ",
    r2_score(Separatrix_ref_lim_test[:, :, 1], Separatrix_lim_test[:, :, 1]),
)

print(
    "R2 train on R coordinate -> ",
    mean_squared_error(
        Separatrix_ref_lim_train[:, :, 0], Separatrix_lim_train[:, :, 0]
    ),
)
print(
    "R2 test on R coordinate -> ",
    mean_squared_error(Separatrix_ref_lim_test[:, :, 0], Separatrix_lim_test[:, :, 0]),
)
print(
    "R2 train on Z coordinate -> ",
    mean_squared_error(
        Separatrix_ref_lim_train[:, :, 1], Separatrix_lim_train[:, :, 1]
    ),
)
print(
    "R2 test on Z coordinate -> ",
    mean_squared_error(Separatrix_ref_lim_test[:, :, 1], Separatrix_lim_test[:, :, 1]),
)


# ### With reference Psi_b
# R2 train on R coordinate ->  0.9933837231689878
# R2 test on R coordinate ->  0.9933686316681145
# R2 train on Z coordinate ->  0.995404419238913
# R2 test on Z coordinate ->  0.9954547583598966

# ### With Psi_b computed by the PlaNet XPlim net
# R2 train on R coordinate ->  0.9954970916737366
# R2 test on R coordinate ->  0.9921427586748792
# R2 train on Z coordinate ->  0.9952510270707666
# R2 test on Z coordinate ->  0.9910695529972325


### Check mismatched equilibria
ind_wrong = np.where(out_clas_train.ravel() != XP_YN_train.ravel())[0]


for i in range(5):
    ind_random = np.random.randint(0, ind_wrong.shape[0], 1)
    XP_YN_train[ind_wrong[ind_random]]
    if XP_YN_train[ind_wrong[ind_random]] == 0:
        title = "Equilibrium is limiter, but classified as diverted"
    else:
        title = "Equilibrium is diverted, but classified as limiter"

    psi_grid = np.squeeze(Psi_train_scaled[ind_wrong[ind_random], :, :])
    psi_b = Psi_b_train_scaled[ind_wrong[ind_random]][0]
    plt.figure()
    plt.contourf(RR_pixels, ZZ_pixels, psi_grid, 25)
    plt.colorbar()
    plt.contour(RR_pixels, ZZ_pixels, psi_grid, [psi_b])
    plt.plot(limiter_geo[:, 0], limiter_geo[:, 1], "k")
    plt.axis("equal")
    plt.title(str(title))
    plt.show()


for i in range(5):
    ind_random = np.random.randint(0, ind_wrong.shape[0], 1)
    XP_YN_train[ind_wrong[ind_random]]

    ind_target = ind_wrong[ind_random][0]
    psi_grid = Psi_train_scaled[ind_target : ind_target + 1, :, :]
    psi_b = Psi_b_train_scaled[ind_target : ind_target + 1]

    separatrix_ref = Separatrix_ref_train[ind_target, :, :]

    if XP_YN_train[ind_wrong[ind_random]] == 0:
        title = "Equilibrium is limiter, but classified as diverted"
        input_ii = [psi_grid, psi_b]
        separatrix_NN = np.squeeze(PlaNet_separatrix_XP.predict(input_ii))

    else:
        title = "Equilibrium is diverted, but classified as limiter"
        input_ii = [psi_grid, psi_b]
        separatrix_NN = np.squeeze(PlaNet_separatrix_lim.predict(input_ii))

    plt.figure()
    plt.contourf(RR_pixels, ZZ_pixels, np.squeeze(psi_grid), 25)
    plt.colorbar()
    # plt.contour(RR_pixels,ZZ_pixels,np.squeeze(psi_grid),[psi_b[0]])
    plt.plot(separatrix_ref[:, 0] + R0, separatrix_ref[:, 1], "k")
    plt.plot(separatrix_NN[:, 0] + R0, separatrix_NN[:, 1], "--g")
    plt.plot(limiter_geo[:, 0], limiter_geo[:, 1], "k")
    plt.axis("equal")
    plt.title(str(title))
    plt.show()
