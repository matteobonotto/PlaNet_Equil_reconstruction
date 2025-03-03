import os
import sys

sys.path.append(os.getcwd())
os.system("export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH")

from helper_functions.general import timer_func
from src.utils import ParseArgs, get_configs, load_data, init_model

from src.train.train import Trainer

num_workers = os.cpu_count() - 1
num_workers = 0

###


@timer_func
def main():

    args = ParseArgs()
    config = get_configs(args)
    train_ds, test_ds = load_data(config)
    model = init_model(test_ds)

    trainer = Trainer(model=model, config=config, train_ds=train_ds)
    trainer.run()


###
if __name__ == "__main__":
    main()
