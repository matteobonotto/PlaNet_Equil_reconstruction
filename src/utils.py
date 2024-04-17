from time import time
from pathlib import Path
# from lightning.pytorch.cli import LightningCLI 

import argparse
import yaml
from src.models.planet import PlaNet_Equil_Neural_Opt
import tensorflow as tf


def timer_func(func):
    """
    Decorator to time function execution

    :param func: function to be timed
    :return: timed function

    """
    def wrapper(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'{func.__name__}() executed in {(t2-t1):.6f}s')
        print('')
        return result
    return wrapper

def touch_dir(dir_path: str) -> None:
    """
    Create dir if not exist

    :param dir_path: directory path

    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)



# class CustomCli(LightningCLI):
#     def add_arguments_to_parser(self, parser) -> None:
#         pass



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
    x_ds, y_ds, _, _, _, _, _ = iter(test_dl).next()
    n_input = x_ds.shape[-1]
    n_grid = y_ds.shape[-1]
    return PlaNet_Equil_Neural_Opt(n_input, n_grid)