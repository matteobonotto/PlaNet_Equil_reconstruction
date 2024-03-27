import os
import sys
sys.path.append(os.getcwd())
os.system('export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH')


# from lightning import Trainer
from helper_functions.general import timer_func
# from lightning.pytorch.callbacks import TQDMProgressBar
# from src.models.pl_model import SimplePytorchLightningModel 
# from src.dataloaders.mnist_dataloader import FashionMnistDataLoader
import tensorflow as tf
import configparser
import argparse
import yaml
from src.models.planet import PlaNet_Equil_Neural_Opt

###

num_workers = os.cpu_count()-1
num_workers = 0


###

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c')
    args,_ = parser.parse_known_args()
    return args

def get_configs(args):
    with open(args.c, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_data(config):
    train_dl = tf.data.Dataset.load(
        config['data']['path']['train']
        ).batch(config['data']['batch_size'])
    test_dl =  tf.data.Dataset.load(
        config['data']['path']['test']
        ).batch(config['data']['batch_size'])
    
    return train_dl, test_dl

def init_model(test_dl):
    x_ds, y_ds, RHS_in_ds = iter(test_dl).next()
    n_input = x_ds.shape[-1]
    n_grid = y_ds.shape[-1]
    return PlaNet_Equil_Neural_Opt(n_input, n_grid)

@tf.function
def train_step(
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

for i in range(4):
    training_time_start = time.time()
    for epoch in range(epochs):
        t_start = time.time()
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
        epoch_time = time.time() - t_start

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

    total_training_time = time.time() - training_time_start
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



@timer_func
def main():

    args = ParseArgs()
    config = get_configs(args)
    train_dl, test_dl = load_data(config)
    model = init_model(test_dl)

    


    



    # Initialize a trainer
    trainer = Trainer(
        accelerator="cuda",
        max_epochs=20,
        callbacks=[TQDMProgressBar(refresh_rate=250)],
        # enable_model_summary=False,
        # barebones=True,
        # enable_checkpointing=False
    )

    trainer.fit(
        SimplePytorchLightningModel(), 
        FashionMnistDataLoader(
            path=path,
            num_workers=num_workers
            )
        )
    

###
if __name__ == "__main__":
    main()















