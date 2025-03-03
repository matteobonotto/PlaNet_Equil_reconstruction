import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from time import time

# from tensorflow.keras import optimizer
import numpy as np
from src.train.utils_train import (
    gauss_kernel,
    fun_GSoperator_NN_conv_smooth_batch_adaptive,
)
from src.utils import touch_dir
from tqdm import tqdm
import wandb
import pickle
import yaml


class Trainer:
    def __init__(self, model, config, train_ds):
        self.model = model
        self.config = config
        self.train_ds = train_ds

        if self.config["train"]["logging"]:
            if self.config["train"]["logging"]["wandb"]:
                self.logger = wandb
                self.logger.init(
                    project=self.config["train"]["logging"]["wandb"]["project"],
                    # id='bnudeqd5',
                    # resume="must"
                )

        if self.config["train"]["lr"]["scheduler"]:
            lr = ExponentialDecay(
                **self.config["train"]["lr"]["scheduler"]["init_args"]
            )
        else:
            lr = self.config["train"]["lr"]
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

        self.InitCheckpointManager()

    def InitCheckpointManager(self):
        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(1), optimizer=self.optimizer, net=self.model
        )
        touch_dir(self.config["train"]["ckpt_path"])
        self.manager = tf.train.CheckpointManager(
            self.ckpt, self.config["train"]["ckpt_path"], max_to_keep=4
        )

        if self.config["train"]["from_ckpt"]:
            self.ckpt.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print("Restored from {}".format(self.manager.latest_checkpoint))
                with open(
                    self.config["train"]["ckpt_path"] + "history_ckpt.pkl", "rb"
                ) as f:
                    self.history = pickle.load(f)
            else:
                print("No checkpoint found, initializing from scratch.")
                self.init_history()
        else:
            print("Initializing from scratch.")
            self.init_history()

    def init_history(self):
        self.history = {
            "loss": [],
            "loss_iter": [],
            "loss_Psi": [],
            "loss_GSO": [],
        }

    def ckeckpoint_here(self):
        save_path = self.manager.save()
        print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
        with open(self.config["train"]["ckpt_path"] + "history_ckpt.pkl", "wb") as f:
            pickle.dump(self.history, f)

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
        learning_rate_Grad=1,
    ):
        loss_Psi = self.loss_fun_MSE(y_ds, predictions)
        # loss_GSO = loss_fun_PDE(y_ds,predictions,RHS_in_ds)
        if self.config["train"]["is_physics_informed"]:
            loss_GSO = self.loss_fun_PDE_adaptive(
                predictions, RHS_in_ds, Laplace_kernel_ds, Df_dr_kernel_ds, RR_ds, ZZ_ds
            )
            return loss_Psi + loss_GSO
        else:
            return loss_Psi

    @tf.function
    def train_step(self, train_ds_i, learning_rate_PDE=1, learning_rate_Grad=1):

        x_ds, y_ds, RHS_in_ds, RR_ds, ZZ_ds, L_ker_ds, Df_ker_ds = train_ds_i
        with tf.GradientTape() as tape:
            predictions = self.model([x_ds, RR_ds, ZZ_ds])
            loss = self.loss_fun_all(
                y_ds,
                predictions,
                RHS_in_ds,
                RR_ds,
                ZZ_ds,
                L_ker_ds,
                Df_ker_ds,
                learning_rate_PDE,
                learning_rate_Grad,
            )

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        # self.loss = loss
        return loss

    @staticmethod
    def loss_fun_MSE(y_ds, predictions):
        loss_Psi = tf.reduce_mean(tf.square(y_ds - tf.squeeze(predictions)))
        return loss_Psi

    def loss_fun_PDE_adaptive(
        self, predictions, RHS_in_ds, Laplace_kernel_ds, Df_dr_kernel_ds, RR_ds, ZZ_ds
    ):
        GS_ope_ref = RHS_in_ds
        GS_ope_ds = fun_GSoperator_NN_conv_smooth_batch_adaptive(
            predictions,
            Laplace_kernel_ds,
            Df_dr_kernel_ds,
            self.model.Gauss_tensor,
            RR_ds,
            ZZ_ds,
        )
        loss_GSO = tf.reduce_mean(tf.square(GS_ope_ref - GS_ope_ds))
        return 0.1 * loss_GSO

    def run(self):
        config = self.config
        epochs = config["train"]["epochs"]
        self.model.Gauss_tensor = gauss_kernel(config["train"]["Gauss_filt_dim"])
        training_time_start = time()
        epoch_start = len(self.history["loss"])
        print("Starting from epoch {}".format(epoch_start))

        for epoch in tqdm(range(epoch_start, epochs), mininterval=5):
            t_start = time()
            loss_iterations = []
            for train_ds_i in tqdm(self.train_ds, mininterval=1, leave=False):

                self.loss = self.train_step(train_ds_i)

                if self.config["train"]["logging"]:
                    self.logger.log({"loss": self.loss.numpy()})
                    self.logger.log(
                        {"lr": self.optimizer.lr((self.optimizer.iterations))}
                    )

                loss_iterations.append(self.loss.numpy())

            self.history["loss_iter"].extend(loss_iterations)
            self.history["loss"].append(np.mean(np.asarray(loss_iterations)))
            epoch_time = time() - t_start

            # check the last batch
            x_ds, y_ds, RHS_in_ds, RR_ds, ZZ_ds, L_ker_ds, Df_ker_ds = train_ds_i
            preds = self.model([x_ds, RR_ds, ZZ_ds])
            loss_Psi = self.loss_fun_MSE(y_ds, preds).numpy()
            loss_GSO = self.loss_fun_PDE_adaptive(
                preds, RHS_in_ds, L_ker_ds, Df_ker_ds, RR_ds, ZZ_ds
            ).numpy()

            self.history["loss_Psi"].append(loss_Psi)
            self.history["loss_GSO"].append(loss_GSO)

            # Print some statistics
            print(
                "epoch {:d}/{:d}, loss: {:2.2e}, loss Psi: {:2.2e}, loss GSO: {:2.2e}, {:d}s/epoch".format(
                    epoch + 1,
                    epochs,
                    self.history["loss"][-1],
                    loss_Psi,
                    loss_GSO,
                    int(epoch_time),
                )
            )

            if self.config["train"]["logging"]:
                for k in ["loss_Psi", "loss_GSO"]:
                    self.logger.log({k: self.history[k][-1]})
                    # print(self.history[k][-1])

            self.ckpt.step.assign_add(1)
            self.ckeckpoint_here()

        total_training_time = time() - training_time_start
        print("")
        print("--> Total training time {:d}s".format(int(total_training_time)))
        print("")
        print("")

        self.save_trained_model()

    def save_trained_model(self):
        if self.config["train"]["trained_model"]:
            filename = (
                self.config["train"]["trained_model"]["dir"]
                + self.config["train"]["trained_model"]["name"]
            )
            print("Saving trained model in {}".format(filename))

            touch_dir(filename)

            # Save model
            self.model.save(filename + "trained_model.keras")

            # Save history
            with open(filename + "history.pkl", "wb") as f:
                pickle.dump(self.history, f)

            # Save configs
            with open(filename + "configs.yaml", "w") as f:
                yaml.dump(self.config, f)
