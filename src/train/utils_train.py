import numpy as np
import tensorflow as tf

def GaussKernel(size=3):
    if size == 3:
        Gaussian_kernel = np.array((
            [1,2,1], [2,4,2], [1,2,1]
            ), dtype='float32')/(16)
    elif size == 5:
        Gaussian_kernel = np.array(
            ([1,4,7,4,1],
            [4,16,26,16,4],
            [7,26,41,26,7],
            [4,16,26,16,4],
            [1,4,7,4,1]
            ), dtype='float32')/(273)
    else:
        raise NotImplementedError 
    Gauss_tensor = tf.expand_dims(
        tf.expand_dims(Gaussian_kernel[::-1, ::-1],axis = -1),axis = -1
        )
    return Gauss_tensor








