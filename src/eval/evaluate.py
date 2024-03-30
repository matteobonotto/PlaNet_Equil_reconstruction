
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import scipy.io as io


class Evaluator():
    def __init__(self,model,config,test_ds):
        self.model = model
        self.config = config
        self.test_ds = test_ds

    def run(self):

        # load grid
        if 'mat' in self.config['data']['path']['grid']:
            geo = io.loadmat(self.config['data']['path']['grid'])

        for test_ds_i in tqdm(self.test_ds,mininterval=1):
            x_ds, y_ds, RHS_in_ds, RR_ds, ZZ_ds, L_ker_ds, Df_ker_ds = test_ds_i

            # compute metrics on poloidal flux

            # compute metrics on GS operator
        






