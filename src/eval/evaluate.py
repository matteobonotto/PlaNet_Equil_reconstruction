
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import scipy.io as io

from tensorflow.keras.metrics import MeanSquaredError
from src.train.utils_train import gauss_kernel, fun_GSoperator_NN_conv_smooth_batch_adaptive


def Mse2DImage(y_true,y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true), axis=(-1,-2))
    

class Evaluator():
    def __init__(self,model,config,test_ds):
        self.model = model
        self.config = config
        self.test_ds = test_ds


    def run(self):
        # load grid
        if 'mat' in self.config['data']['path']['grid']:
            x_ds, y_ds, RHS_in_ds = next(iter(self.test_ds))
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
            # x_ds, y_ds, RHS_in_ds, RR_ds, ZZ_ds, L_ker_ds, Df_ker_ds  = test_ds_i
            x_ds, y_ds, RHS_in_ds  = test_ds_i

            # MSE on poloidal flux
            pred = self.model([x_ds,RR_ds,ZZ_ds])[...,0]
            mse_flux.append(Mse2DImage(y_ds,pred))

            # MSE on GS operator
            # GS_ope_ds = fun_GSoperator_NN_conv_smooth_batch_adaptive(
            #     pred,
            #     L_ker_ds,
            #     Df_ker_ds,
            #     self.model.Gauss_tensor,
            #     RR_ds,
            #     ZZ_ds)
            # mse_GSope.append(Mse2DImage(RHS_in_ds,GS_ope_ds))

        self.mse_flux = np.array(mse_flux)
        # self.mse_GSope = np.array(mse_GSope)
            








