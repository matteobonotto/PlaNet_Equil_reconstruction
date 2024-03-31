
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import scipy.io as io

from tensorflow.keras.metrics import MeanSquaredError

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
            geo = io.loadmat(self.config['data']['path']['grid'])
            RR_ds,ZZ_ds = geo['RR_pixels'], geo['ZZ_pixels']
            RR_ds = tf.tile(
                tf.expand_dims(RR_ds, axis=0), 
                (x_ds.shape[0],1,1))
            ZZ_ds = tf.tile(
                tf.expand_dims(ZZ_ds, axis=0), 
                (x_ds.shape[0],1,1))
            
        mse = []
        for test_ds_i in tqdm(self.test_ds,mininterval=1):
            x_ds, y_ds, RHS_in_ds = test_ds_i
            pred = self.model([x_ds,RR_ds,ZZ_ds])[...,0]
            mse.append(Mse2DImage(y_ds,pred))








