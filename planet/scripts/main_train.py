
import numpy as np
import tensorflow as tf
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import scipy
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
from copy import deepcopy

from typing import Tuple, List

def fun_contourf_sol(z,RR,ZZ):
    plt.figure()
    plt.contourf(RR, ZZ, z, 20)
    plt.axis('equal')
    plt.colorbar()
    plt.show()
    return


def fun_contour_sol(z,RR,ZZ):
    plt.figure()
    plt.contour(RR, ZZ, z, 20)
    plt.axis('equal')
    plt.colorbar()
    plt.show()
    return


def fun_contour_compare_sol(z_ref,z,RR,ZZ):
    l1 = mlines.Line2D([], [], label='DNN')
    l2 = mlines.Line2D([], [], color='black', label='FRIDA')

    plt.figure()
    plt.contour(RR, ZZ, z, 10)
    plt.colorbar()
    plt.contour(RR, ZZ, z_ref, 10, colors = 'black', linestyles = 'dashed')
    plt.legend(handles=[l1, l2])
    plt.axis('equal')
    plt.show()
    return



class TrainableSwish(nn.Module):
    def __init__(self, beta:float = 1.):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x:Tensor) -> Tensor:
        return swish(x, self.beta)


def swish(x:Tensor, beta:float=1.0) -> Tensor:
    return x * F.sigmoid(beta * x)



class Conv2dNornAct(nn.Module):
    def __init__(self, beta:float = 1.):
        super().__init__()
        self.conv2d = nn.Conv2d()
        self.norm = nn.BatchNorm2d()
        self.act = TrainableSwish()

    def forward(self, x:Tensor) -> Tensor:
        # x = layers.Conv2D(filters=filters ,
        #                 kernel_size=kernel_size,
        #                 strides=1,
        #                 padding='same',
        #                 dtype = DTYPE)(x)
        # x = layers.BatchNormalization()(x)
        # # x = tf.keras.activations.relu(x)
        # # x = tf.keras.activations.tanh(x) if activation == 'tanh' else tf.keras.activations.swish(x)
        # # x = tf.keras.activations.tanh(x) if activation == 'tanh' else tf.keras.activations.swish(x)
        # x = Swish(beta=1.0, trainable=True)(x)
        return self.act(self.norm(self.conv2d(x)))


class TrunkNet(nn.Module):
    def __init__(self, gridsize: Tuple[int,int]):
        super().__init__()
        self.norm_r = nn.BatchNorm2d()
        self.branch_r = [(Conv2dNornAct(), nn.MaxPool2d()) for _ in 3]
        self.norm_z = nn.BatchNorm2d()
        self.branch_z = [(Conv2dNornAct(), nn.MaxPool2d()) for _ in 3]
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear()
        self.act = TrainableSwish()
        self.linear_2 = nn.Linear()


    def forward(self, x:Tuple[Tensor, Tensor]):
        x_r, x_z = x

        # branch for x_r
        x_r = self.norm_r(x)
        for layer, maxpool2d in self.branch_r:
            x_r = maxpool2d(layer(x_r))

        # branch for x_z
        x_z = self.norm_z(x)
        for layer, maxpool2d in self.branch_z:
            x_z = maxpool2d(layer(x_z))

        # concatenate branches and output
        x = torch.cat((x_r, x_z), dim=-1)
        x = self.flatten(x)
        x = self.act(self.linear(x))
        x = self.linear_2(x)
        return x
        
        # # x_r = input_query_RR
        # x_r = layers.BatchNormalization()(input_query_RR)
        # for i in range(3):
        #     # x_r = input_query_RR if i == 0 else x_r
        #     x_r = conv2D_Norm_activation(x_r,filters=(i+1)*8,kernel_size=(3, 3))
        #     x_r = layers.MaxPooling2D(pool_size=(2,2))(x_r)

        # # x_z = input_query_ZZ
        # x_z = layers.BatchNormalization()(input_query_ZZ)
        # for i in range(3):
        #     # x_z = input_query_ZZ if i == 0 else x_z
        #     x_z = conv2D_Norm_activation(x_z,filters=(i+1)*8,kernel_size=(3, 3))
        #     x_z = layers.MaxPooling2D(pool_size=(2,2))(x_z)

        # out_trunk = layers.Concatenate()([x_r,x_z])
        # out_trunk = layers.Flatten()(out_trunk)
        # out_trunk = layers.Dense(128,
        #                 #  activation=tf.keras.activations.get('silu'),
        #                 #  kernel_initializer='he_normal',
        #                 dtype = DTYPE)(out_trunk)
        # out_trunk = Swish(beta=1.0, trainable=True)(out_trunk)

        # # for i in range(2):
        # #     out_trunk = layers.Dense(128,
        # #                  activation=tf.keras.activations.get('gelu'),
        # #                  kernel_initializer='he_normal',
        # #                  dtype = DTYPE)(out_trunk)

        # # x_r = layers.Flatten()(input_query_RR)
        # # x_r = layers.BatchNormalization()(x_r)
        # # for i in range(3):
        # #     x_r = layers.Dense(64,
        # #                  activation=tf.keras.activations.get('tanh'),
        # #                  kernel_initializer='he_normal',
        # #                  dtype = DTYPE)(x_r)

        # # x_z = layers.Flatten()(input_query_ZZ)
        # # x_z = layers.BatchNormalization()(x_z)
        # # for i in range(3):
        # #     x_z = layers.Dense(64,
        # #                  activation=tf.keras.activations.get('tanh'),
        # #                  kernel_initializer='he_normal',
        # #                  dtype = DTYPE)(x_z)

        # # out_trunk = layers.Concatenate()([x_r,x_z])

        # for i in range(2):
        # out_trunk = layers.Dense(64,
        #                 #  activation=tf.keras.activations.get('silu'),
        #                 #  kernel_initializer='he_normal',
        #                 dtype = DTYPE)(out_trunk)
        # out_trunk = Swish(beta=1.0, trainable=True)(out_trunk)        


class BranchNet(nn.Module):
    def __init__(self, gridsize: Tuple[int,int]):
        super().__init__()
        nx,ny = gridsize
        self.linear_1 = nn.Linear()
        self.norm_1 = nn.BatchNorm1d()
        self.act_1 = TrainableSwish(beta=1.0, trainable=True)
        self.linear_2 = nn.Linear()
        self.norm_2 = nn.BatchNorm1d()
        self.act_2 = TrainableSwish(beta=1.0, trainable=True)
        self.linear_3 = nn.Linear()

    def forward(self):
        x = self.act_1(self.norm_1(self.linear_1(x)))
        x = self.act_2(self.norm_2(self.linear_2(x)))
        x = self.linear_3(x)

        return x

        # pass

        # x = layers.Dense(256,
        #                 #  activation=tf.keras.activations.get('silu'),
        #                 #  kernel_initializer='he_normal',
        #                 #  kernel_regularizer=tf.keras.regularizers.L2(0.005),
        #                 dtype = DTYPE)(input_fun)
        # x = Swish(beta=1.0, trainable=True)(x)

        # x = layers.Dense(128,
        #                 #  activation=tf.keras.activations.get('silu'),
        #                 #  kernel_initializer='he_normal',
        #                 #  kernel_regularizer=tf.keras.regularizers.L2(0.005),
        #                 dtype = DTYPE)(x)
        # x = Swish(beta=1.0, trainable=True)(x)

        # x = layers.Dense(64,
        #                 #  activation=tf.keras.activations.get('silu'),
        #                 #  kernel_initializer='he_normal',
        #                 #  kernel_regularizer=tf.keras.regularizers.L2(0.005),
        #                 dtype = DTYPE)(x)
        # out_branch = Swish(beta=1.0, trainable=True)(x)


class Deconder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear

    def forward(self):
        
        # Multiply layer
        out_multiply = layers.Multiply(name='Multiply')([out_branch,out_trunk])


        # conv2d-based decoder
        x_dec = layers.Dense(neuron_FC,
                        #  activation=tf.keras.activations.get('silu'),
                        #  kernel_initializer='he_normal',
                        #  kernel_regularizer=tf.keras.regularizers.L2(0.005),
                        dtype = DTYPE)(out_multiply)
        x_dec = Swish(beta=1.0, trainable=True)(x_dec)

        x_dec = layers.Reshape(target_shape=(n_w, n_h, n_c))(x_dec)

        x_dec = layers.UpSampling2D(size = (2,2),
                                interpolation = interpolation)(x_dec)
        x_dec = conv2D_Norm_activation(x_dec,filters=32,kernel_size=(3, 3))


        x_dec = layers.UpSampling2D(size = (2,2),
                                interpolation = interpolation)(x_dec)
        x_dec = conv2D_Norm_activation(x_dec,filters=16,kernel_size=(3, 3))


        x_dec = layers.UpSampling2D(size = (2,2),
                                interpolation = interpolation)(x_dec)
        x_dec = conv2D_Norm_activation(x_dec,filters=8,kernel_size=(3, 3))

        out_grid = layers.Conv2D(filters=1,
                        kernel_size=(1, 1),
                        strides=1,
                        padding='same',
                        activation='linear',
                        kernel_initializer='he_normal',
                        bias_initializer='zeros',
                        dtype = DTYPE)(x_dec)


        outputs = out_grid


class PleNetCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = TrunkNet()
        self.branch = BranchNet()
        self.decoder = Deconder()

    def forward(self):
        pass




if __name__ == "__main__":
    train_ds_1 = tf.data.Dataset.load('./data/tf_Dataset_NeuralOpt_all_domain_only_32x32.data')
    train_ds_2 = tf.data.Dataset.load('./data/tf_Dataset_NeuralOpt_super_res_only_32x32.data')
    # train_ds_1 = tf.data.Dataset.load('./gdrive/MyDrive/Colab_Notebooks/tf_Datasets/tf_Dataset_NeuralOpt_all_domain_only_64x64.data')
    # train_ds_2 = tf.data.Dataset.load('./gdrive/MyDrive/Colab_Notebooks/tf_Datasets/tf_Dataset_NeuralOpt_super_res_only_64x64.data')
    dict_geo = scipy.io.loadmat('./data/data_geo_Dataset_NeuralOpt_super_res_32x32.mat')

    train_ds = train_ds_1.concatenate(train_ds_2)
    train_ds = train_ds.shuffle(42)
    # train_ds = train_ds.batch(1024)

    x_ds, y_ds, RHS_in_ds, RR_ds, ZZ_ds, Laplace_kernel_ds, Df_dr_kernel_ds = iter(train_ds).next()

    print(f'batch size: {x_ds.shape[0]}')
    print(train_ds.cardinality().numpy()*x_ds.shape[0])

    n_grid = RR_ds.shape[1]
    n_output = y_ds.shape[1]
    n_input = x_ds.shape[1]

    RR_pixels = dict_geo['RR_pixels']
    ZZ_pixels = dict_geo['ZZ_pixels']

    RHS_i = RHS_in_ds[-1,:,:].numpy()
    psi_i = y_ds[-1,:,:].numpy()



