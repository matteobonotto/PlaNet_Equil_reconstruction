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
from src.train.train import train
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




@timer_func
def main():

    args = ParseArgs()
    config = get_configs(args)
    train_dl, test_dl = load_data(config)
    model = init_model(test_dl)

    
    model = train(model)

    



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















