import keras
from keras.layers import Layer
from keras import backend as K

# import keras.src.backend.common as K
import keras.backend as K

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import mixed_precision

# import numpy as np
# import keras
# from keras import ops
# from keras import layers

"""
###############################################################################
### Physics-Informed Neural Operator
"""
DTYPE = "float32"


def swish(x, beta=1.0):
    return x * K.sigmoid(beta * x)


# class Swish(keras.layers.Layer):
#     def __init__(self,beta=1):
#         super().__init__()
#         self.beta_factor = self.add_weight(
#             shape=(1,),
#             initializer=keras.initializers.RandomNormal(mean=beta, stddev=0.05, seed=None),
#             trainable=True,
#         )

#     def call(self, inputs):
#         return inputs * K.sigmoid(self.beta_factor * inputs)


class Swish(Layer):

    def __init__(self, beta=1.0, trainable=False, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta
        self.trainable = trainable

    def build(self, input_shape):
        self.beta_factor = K.variable(self.beta, dtype=K.floatx(), name="beta_factor")
        if self.trainable:
            self._trainable_weights.append(self.beta_factor)

        super(Swish, self).build(input_shape)

    def call(self, inputs, mask=None):
        return swish(inputs, self.beta_factor)

    def get_config(self):
        config = {
            "beta": self.get_weights()[0] if self.trainable else self.beta,
            "trainable": self.trainable,
        }
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def conv2D_Norm_activation(x, filters, kernel_size, activation="swish"):
    x = layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=1, padding="same", dtype=DTYPE
    )(x)
    x = layers.BatchNormalization()(x)
    # x = tf.keras.activations.relu(x)
    # x = tf.keras.activations.tanh(x) if activation == 'tanh' else tf.keras.activations.swish(x)
    # x = tf.keras.activations.tanh(x) if activation == 'tanh' else tf.keras.activations.swish(x)
    x = Swish(beta=1.0, trainable=True)(x)
    return x


def conv2D_Norm_activation(x, filters, kernel_size, activation="swish"):
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        # activation='silu',
        dtype=DTYPE,
    )(x)
    x = Swish(beta=1.0, trainable=True)(x)
    # x = tf.keras.activations.tanh(x) if activation == 'tanh' else tf.keras.activations.swish(x)
    return x


def PlaNet_Equil_Neural_Opt(n_input, n_grid):
    input_shape_fun = n_input
    input_query_r = n_grid
    input_query_z = n_grid
    neuron_FC = 2048
    n_w = 8 if n_grid == 64 else 4
    n_h = 8 if n_grid == 64 else 4
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
        256,
        #  activation=tf.keras.activations.get('silu'),
        #  kernel_initializer='he_normal',
        #  kernel_regularizer=tf.keras.regularizers.L2(0.005),
        dtype=DTYPE,
    )(input_fun)
    x = Swish(beta=1.0, trainable=True)(x)

    x = layers.Dense(
        128,
        #  activation=tf.keras.activations.get('silu'),
        #  kernel_initializer='he_normal',
        #  kernel_regularizer=tf.keras.regularizers.L2(0.005),
        dtype=DTYPE,
    )(x)
    x = Swish(beta=1.0, trainable=True)(x)

    x = layers.Dense(
        64,
        #  activation=tf.keras.activations.get('silu'),
        #  kernel_initializer='he_normal',
        #  kernel_regularizer=tf.keras.regularizers.L2(0.005),
        dtype=DTYPE,
    )(x)
    out_branch = Swish(beta=1.0, trainable=True)(x)

    # out_branch = layers.Dense(64,
    #                  activation=tf.keras.activations.get('silu'),
    #                  kernel_initializer='he_normal',
    #                 #  kernel_regularizer=tf.keras.regularizers.L2(0.005),
    #                  dtype = DTYPE)(x)

    # Trunk net
    # x_r = input_query_RR
    x_r = layers.BatchNormalization()(input_query_RR)
    for i in range(3):
        # x_r = input_query_RR if i == 0 else x_r
        x_r = conv2D_Norm_activation(x_r, filters=(i + 1) * 8, kernel_size=(3, 3))
        x_r = layers.MaxPooling2D(pool_size=(2, 2))(x_r)

    # x_z = input_query_ZZ
    x_z = layers.BatchNormalization()(input_query_ZZ)
    for i in range(3):
        # x_z = input_query_ZZ if i == 0 else x_z
        x_z = conv2D_Norm_activation(x_z, filters=(i + 1) * 8, kernel_size=(3, 3))
        x_z = layers.MaxPooling2D(pool_size=(2, 2))(x_z)

    out_trunk = layers.Concatenate()([x_r, x_z])
    out_trunk = layers.Flatten()(out_trunk)
    out_trunk = layers.Dense(
        128,
        #  activation=tf.keras.activations.get('silu'),
        #  kernel_initializer='he_normal',
        dtype=DTYPE,
    )(out_trunk)
    out_trunk = Swish(beta=1.0, trainable=True)(out_trunk)

    # for i in range(2):
    #     out_trunk = layers.Dense(128,
    #                  activation=tf.keras.activations.get('gelu'),
    #                  kernel_initializer='he_normal',
    #                  dtype = DTYPE)(out_trunk)

    # x_r = layers.Flatten()(input_query_RR)
    # x_r = layers.BatchNormalization()(x_r)
    # for i in range(3):
    #     x_r = layers.Dense(64,
    #                  activation=tf.keras.activations.get('tanh'),
    #                  kernel_initializer='he_normal',
    #                  dtype = DTYPE)(x_r)

    # x_z = layers.Flatten()(input_query_ZZ)
    # x_z = layers.BatchNormalization()(x_z)
    # for i in range(3):
    #     x_z = layers.Dense(64,
    #                  activation=tf.keras.activations.get('tanh'),
    #                  kernel_initializer='he_normal',
    #                  dtype = DTYPE)(x_z)

    # out_trunk = layers.Concatenate()([x_r,x_z])

    for i in range(2):
        out_trunk = layers.Dense(
            64,
            #  activation=tf.keras.activations.get('silu'),
            #  kernel_initializer='he_normal',
            dtype=DTYPE,
        )(out_trunk)
        out_trunk = Swish(beta=1.0, trainable=True)(out_trunk)

    # Multiply layer
    out_multiply = layers.Multiply(name="Multiply")([out_branch, out_trunk])

    # conv2d-based decoder
    x_dec = layers.Dense(
        neuron_FC,
        #  activation=tf.keras.activations.get('silu'),
        #  kernel_initializer='he_normal',
        #  kernel_regularizer=tf.keras.regularizers.L2(0.005),
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

    # x = layers.Resizing(height = y_train.shape[1],width = y_train.shape[2],dtype = DTYPE)(x)
    # outputs = x

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
    )
    model.compile(optimizer="adam", loss="mse", run_eagerly=False)

    return model


# model = PlaNet_Equil_Neural_Opt()
# model.summary()
# model.save('PlaNet_Equil_Neural_Opt_enc_dec.keras')
