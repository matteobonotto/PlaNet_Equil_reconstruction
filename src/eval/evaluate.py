
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import scipy.io as io

from tensorflow.keras.metrics import MeanSquaredError


class Evaluator():
    def __init__(self,model,config,test_ds):
        self.model = model
        self.config = config
        self.test_ds = test_ds

    def run(self):

        # load grid
        if 'mat' in self.config['data']['path']['grid']:
            geo = io.loadmat(self.config['data']['path']['grid'])
            RR_ds,ZZ_ds = geo['RR_pixels'], geo['ZZ_pixels']
            RR_ds = tf.tile(
                tf.expand_dims(RR_ds, axis=0), 
                (x_ds.shape[0],1,1))
            ZZ_ds = tf.tile(
                tf.expand_dims(ZZ_ds, axis=0), 
                (x_ds.shape[0],1,1))
            
        MSE = MeanSquaredError()

        mse = []

        for test_ds_i in tqdm(self.test_ds,mininterval=1):
            x_ds, y_ds, RHS_in_ds = test_ds_i

            pred = self.model.predict([x_ds,RR_ds,ZZ_ds])

            qq = tf.reduce_mean(tf.square(pred - y_ds))

            # compute metrics on poloidal flux
            for i in range(pred.shape[0]):
                mse.append(MSE(y_ds[i,...],pred[i,...,0]))

            # compute metrics on GS operator
        






