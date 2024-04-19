
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import scipy.io as io

from tensorflow.keras.metrics import MeanSquaredError
from src.train.utils_train import gauss_kernel, fun_GSoperator_NN_conv_smooth_batch_adaptive

DTYPE = 'float32'


def Mse2DImage(y_true,y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true), axis=(-1,-2))

def compute_GSO_kernels(RR_ds, ZZ_ds):

    Laplace_kernel = np.zeros((RR_ds.shape[0],3,3),dtype=DTYPE)
    Df_dr_kernel = np.zeros((RR_ds.shape[0],3,3),dtype=DTYPE)

    for i in tqdm(range(ZZ_ds.shape[0])):
        hr = RR_ds[i,1,2] - RR_ds[i,1,1]
        hz = ZZ_ds[i,2,1] - ZZ_ds[i,1,1] 
        alfa = -2*(hr**2 + hz**2)
        Laplace_kernel_i = np.array(([0, hr**2/alfa, 0], [hz**2/alfa, 1, hz**2/alfa], [0, hr**2/alfa, 0]))
        Df_dr_kernel_i = np.array(([0, 0, 0], [+1, 0, -1], [0, 0, 0]))/(2*hr*alfa)*(hr**2*hz**2)

        Laplace_kernel[i,:,:] = Laplace_kernel_i
        Df_dr_kernel[i,:,:] = Df_dr_kernel_i
    return Laplace_kernel, Df_dr_kernel



class Evaluator():
    def __init__(self,model,config,test_ds):
        self.model = model
        self.config = config
        self.test_ds = test_ds


    def run(self):
        # load grid
        if 'mat' in self.config['data']['path']['grid']:
            x_ds, y_ds, RHS_in_ds, _, _, _, _ = next(iter(self.test_ds))
            geo = io.loadmat(self.config['data']['path']['grid'])
            RR_ds,ZZ_ds = geo['RR_pixels'], geo['ZZ_pixels']
            RR_ds = tf.tile(
                tf.expand_dims(RR_ds, axis=0), 
                (x_ds.shape[0],1,1))
            ZZ_ds = tf.tile(
                tf.expand_dims(ZZ_ds, axis=0), 
                (x_ds.shape[0],1,1))
            
        mse_flux = []
        mse_GSope = []
        for test_ds_i in tqdm(self.test_ds,mininterval=1):
            x_ds, y_ds, RHS_in_ds, RR_ds, ZZ_ds, L_ker_ds, Df_ker_ds  = test_ds_i

            # MSE on poloidal flux
            pred = self.model([x_ds,RR_ds,ZZ_ds])
            mse_flux.append(Mse2DImage(y_ds,pred[...,0]))

            # MSE on GS operator
            GS_ope_ds = fun_GSoperator_NN_conv_smooth_batch_adaptive(
                pred,
                L_ker_ds,
                Df_ker_ds,
                gauss_kernel(self.config['train']['Gauss_filt_dim']),
                RR_ds,
                ZZ_ds)
            mse_GSope.append(Mse2DImage(RHS_in_ds,GS_ope_ds))

        self.mse_flux = np.array(mse_flux)
        self.mse_GSope = np.array(mse_GSope)
            








