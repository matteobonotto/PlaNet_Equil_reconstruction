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
from src.train.train import Trainer
from src.eval.evaluate import Evaluator

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
    train_ds = tf.data.Dataset.load(
        config['data']['path']['train']
        ).batch(config['data']['batch_size'],drop_remainder=True).shuffle(
            buffer_size = 70000,
            seed=42,
            reshuffle_each_iteration=True
            )
    test_ds =  tf.data.Dataset.load(
        config['data']['path']['test']
        ).batch(config['data']['batch_size'],drop_remainder=True)
    
    return train_ds, test_ds

def init_model(test_dl):
    x_ds, y_ds, _ = iter(test_dl).next()
    n_input = x_ds.shape[-1]
    n_grid = y_ds.shape[-1]
    return PlaNet_Equil_Neural_Opt(n_input, n_grid)




@timer_func
def main():

    args = ParseArgs()
    config = get_configs(args)
    train_ds, test_ds = load_data(config)
    model = init_model(test_ds)

    trainer = Trainer(
        model=model,
        config=config,
        train_ds = train_ds
    )
    trainer.run()

    # evaluator = Evaluator(
    #     model=model,
    #     config=config,
    #     test_ds = test_ds
    # )
    # evaluator.run()


    

###
if __name__ == "__main__":
    main()















