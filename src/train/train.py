

import tensorflow as tf
from time import time


@tf.function
def train_step(
    model,
    x_ds,
    y_ds,
    RHS_in_ds,
    RR_ds,
    ZZ_ds,
    Laplace_kernel_ds,
    Df_dr_kernel_ds,
    learning_rate_PDE=1,
    learning_rate_Grad=1):

  with tf.GradientTape() as tape:
      predictions = model([x_ds,RR_ds,ZZ_ds])
      loss = loss_fun_all(
          y_ds,
          predictions,
          RHS_in_ds,
          RR_ds,
          ZZ_ds,
          Laplace_kernel_ds,
          Df_dr_kernel_ds,
          learning_rate_PDE,
          learning_rate_Grad)

  gradients = tape.gradient(loss,model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss


print_every_epoch = 25
epochs = 500
optimizer = tf.keras.optimizers.Adam() # poor performances
optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
# optimizer = tf.keras.optimizers.Adam(learning_rate=.0005)


def trainer(config):
    for i in range(4):
        training_time_start = time()
        for epoch in range(epochs):
            t_start = time()
            loss_epoch = []
            for x_ds, y_ds, RHS_in_ds, RR_ds, ZZ_ds, L_ker_ds, Df_ker_ds in train_ds:

                loss = train_step(
                    x_ds,
                    y_ds,
                    RHS_in_ds,
                    RR_ds,
                    ZZ_ds,
                    L_ker_ds,
                    Df_ker_ds,
                    learning_rate_PDE,
                    learning_rate_Grad)

                loss_epoch.append(loss)

            loss = np.mean(np.asarray(loss_epoch))
            epoch_time = time() - t_start

            history.append(loss)
            ckpt.step.assign_add(1)
            if epoch % print_every_epoch == 0:
                print('epoch {:d}, of {:d}, loss = {:2.2e}, {:d}s/epoch'.format(
                    epoch+1,
                    epochs,
                    loss,
                    int(epoch_time)))
                preds = model([x_ds, RR_ds, ZZ_ds])
                print(loss_fun_MSE(y_ds, preds).numpy(),
                    loss_fun_PDE_adaptive(
                        preds,
                        RHS_in_ds,
                        L_ker_ds,
                        Df_ker_ds,
                        RR_ds,
                        ZZ_ds).numpy())

                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

        total_training_time = time() - training_time_start
        print('')
        print('--> Total training time {:d}s'.format(int(total_training_time)))
        print('')
        print('')

        dictionary = {'history': history}
        history_name = ('/content/gdrive/MyDrive/history_PlaNet_NeuralOp_test_{:d}epochs.h5'.format(len(history)))
        model_name = ('/content/gdrive/MyDrive/model_PlaNet_NeuralOp_test_{:d}epochs.h5'.format(len(history)))
        scipy.io.savemat(history_name,dictionary)
        model.save(model_name)

    plt.figure()
    plt.plot(np.arange(0,len(history)),history, marker = '.')
    plt.yscale('log')

    model.save('/content/gdrive/MyDrive/tmp_model.h5')



















