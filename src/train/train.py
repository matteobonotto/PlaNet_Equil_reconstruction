

import tensorflow as tf
from time import time
# from tensorflow.keras import optimizer
import numpy as np
from src.train.utils_train import GaussKernel
from src.utils import touch_dir
from tqdm import tqdm
import wandb

wandb.init(project = "PlaNet")
Gauss_tensor = GaussKernel()



class Trainer():
    def __init__(self,model,config,train_ds):
        self.model = model
        self.config = config
        self.train_ds = train_ds
        self.InitCheckpointManager()


    def InitCheckpointManager(self):
        self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
        touch_dir(self.config['train']['ckpt_path'])
        self.manager = tf.train.CheckpointManager(
            self.ckpt, 
            self.config['train']['ckpt_path'], 
            max_to_keep=4)
        

    def loss_fun_all(
            self,
            y_ds,
            predictions,
            RHS_in_ds,
            RR_ds,
            ZZ_ds,
            Laplace_kernel_ds,
            Df_dr_kernel_ds,
            learning_rate_PDE=1,
            learning_rate_Grad=1):
        loss_MSE = self.loss_fun_MSE(y_ds, predictions)
        # loss_PDE = loss_fun_PDE(y_ds,predictions,RHS_in_ds)
        if self.config['train']['is_physics_informed']:
            loss_PDE = self.loss_fun_PDE_adaptive(
            predictions,
            RHS_in_ds,
            Laplace_kernel_ds,
            Df_dr_kernel_ds,
            RR_ds,
            ZZ_ds)
            return loss_MSE + loss_PDE
        else:
            return loss_MSE


    @tf.function
    def train_step(
        self,
        train_ds_i,
        learning_rate_PDE=1,
        learning_rate_Grad=1):

        x_ds, y_ds, RHS_in_ds, RR_ds, ZZ_ds, L_ker_ds, Df_ker_ds = train_ds_i
        with tf.GradientTape() as tape:
            predictions = self.model([x_ds,RR_ds,ZZ_ds])
            loss = self.loss_fun_all(
                y_ds,
                predictions,
                RHS_in_ds,
                RR_ds,
                ZZ_ds,
                L_ker_ds,
                Df_ker_ds,
                learning_rate_PDE,
                learning_rate_Grad)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
    

    @staticmethod
    def fun_GSoperator_NN_conv_smooth_batch_adaptive(
        f,
        Laplace_kernel_ds,
        Df_dr_kernel_ds,
        RR_ds,
        ZZ_ds):

        f = tf.transpose(f,[3,1,2,0])
        Lpsi = tf.nn.depthwise_conv2d(
            f,
            tf.transpose(tf.expand_dims(Laplace_kernel_ds, axis=-1), [1,2,0,3]),
            strides=[1, 1, 1, 1],
            padding='VALID')
        Lpsi = tf.transpose(Lpsi,[3,1,2,0]) # no need to be transposed becaused Laplacian filter is left/rigth symmetric
        Dpsi_dr = tf.nn.depthwise_conv2d(
            f,
            tf.transpose(tf.expand_dims(Df_dr_kernel_ds, axis=-1), [1,2,0,3]),
            strides=[1, 1, 1, 1],
            padding='VALID')
        Dpsi_dr = - Dpsi_dr # necessary because nn.depthwise_conv2d filters has to be transposed to perform real convolution (here [+h 0 -h] -> [-h 0 +h])
        Dpsi_dr = tf.transpose(Dpsi_dr,[3,1,2,0])
        RR_in = tf.expand_dims(RR_ds[:,1:-1,1:-1],axis=-1)
        Dpsi_dr = tf.math.divide(Dpsi_dr,RR_in)

        GS_ope = Lpsi - Dpsi_dr

        hr = RR_ds[:,1,2] - RR_ds[:,1,1]
        hz = ZZ_ds[:,2,1] - ZZ_ds[:,1,1]
        alfa = -2*(hr**2 + hz**2)

        hr = tf.expand_dims(tf.expand_dims(tf.expand_dims(hr,axis=-1),axis=-1),axis=-1)
        hz = tf.expand_dims(tf.expand_dims(tf.expand_dims(hz,axis=-1),axis=-1),axis=-1)
        alfa = tf.expand_dims(tf.expand_dims(tf.expand_dims(alfa,axis=-1),axis=-1),axis=-1)

        GS_ope = GS_ope*alfa/(hr**2*hz**2)
        # GS_ope[0,:10,0]

        GS_ope = tf.nn.conv2d(GS_ope,Gauss_tensor,strides=[1, 1, 1, 1],padding='SAME')
        GS_ope = tf.squeeze(GS_ope,axis = -1)

        return GS_ope
    

    @staticmethod
    def loss_fun_MSE(y_ds, predictions):
        loss_MSE = tf.reduce_mean(tf.square(y_ds - tf.squeeze(predictions)))
        return loss_MSE
    
    def loss_fun_PDE_adaptive(
            self,
            predictions,
            RHS_in_ds,
            Laplace_kernel_ds,
            Df_dr_kernel_ds,
            RR_ds,
            ZZ_ds):
        GS_ope_ref = RHS_in_ds
        GS_ope_ds = self.fun_GSoperator_NN_conv_smooth_batch_adaptive(
            predictions,
            Laplace_kernel_ds,
            Df_dr_kernel_ds,
            RR_ds,
            ZZ_ds)
        loss_PDE = tf.reduce_mean(tf.square(GS_ope_ref - GS_ope_ds))
        return .1*loss_PDE


    def run(self):
        learning_rate_MSE  = 1
        learning_rate_Grad = 1
        learning_rate_PDE  = 1
        config = self.config
        epochs = config['train']['epochs']
        # for i in range(config['train']['epochs']):
        training_time_start = time()
        self.history = []
        for epoch in tqdm(range(epochs),mininterval=5):
            t_start = time()
            loss_epoch = []
            for train_ds_i in tqdm(
                self.train_ds,
                mininterval=1,
                leave=False):

                loss = self.train_step(
                    train_ds_i,
                    learning_rate_PDE,
                    learning_rate_Grad)
                
                wandb.log({"loss": loss.numpy()})

                loss_epoch.append(loss.numpy())

            loss = np.mean(np.asarray(loss_epoch))
            epoch_time = time() - t_start

            self.history.append(loss)
            self.ckpt.step.assign_add(1)
            if epoch % config['train']['print_every_epoch'] == 0:
                print('epoch {:d}, of {:d}, loss = {:2.2e}, {:d}s/epoch'.format(
                    epoch+1,
                    epochs,
                    loss,
                    int(epoch_time)))
                x_ds, y_ds, RHS_in_ds, RR_ds, ZZ_ds, L_ker_ds, Df_ker_ds = train_ds_i
                preds = self.model([x_ds, RR_ds, ZZ_ds])
                loss_fun_MSE = self.loss_fun_MSE(y_ds, preds).numpy()
                loss_fun_PDE_adaptive = self.loss_fun_PDE_adaptive(
                        preds,
                        RHS_in_ds,
                        L_ker_ds,
                        Df_ker_ds,
                        RR_ds,
                        ZZ_ds).numpy()
                wandb.log({"loss_fun_MSE": loss_fun_MSE})
                wandb.log({"loss_fun_PDE_adaptive": loss_fun_PDE_adaptive})
                print(loss_fun_MSE, loss_fun_PDE_adaptive)

                save_path = self.manager.save()
                print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

        total_training_time = time() - training_time_start
        print('')
        print('--> Total training time {:d}s'.format(int(total_training_time)))
        print('')
        print('')

        # dictionary = {'history': self.history}
        # history_name = ('/content/gdrive/MyDrive/history_PlaNet_NeuralOp_test_{:d}epochs.h5'.format(len(self.history)))
        # model_name = ('/content/gdrive/MyDrive/model_PlaNet_NeuralOp_test_{:d}epochs.h5'.format(len(self.history)))
        # scipy.io.savemat(history_name,dictionary)
        # self.model.save(model_name)

        # plt.figure()
        # plt.plot(np.arange(0,len(history)),history, marker = '.')
        # plt.yscale('log')

        # model.save('/content/gdrive/MyDrive/tmp_model.h5')



















