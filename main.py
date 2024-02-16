import os
import sys
from lightning import Trainer
from src.utils import timer_func
from lightning.pytorch.callbacks import TQDMProgressBar
from src.models.pl_model import SimplePytorchLightningModel 
from src.dataloaders.mnist_dataloader import FashionMnistDataLoader


###
sys.path.append(os.getcwd())
os.system('export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH')

num_workers = os.cpu_count()-1
num_workers = 0


###
@timer_func
def main():
    path = r'./data/fashion-mnist/fashion-mnist_train.csv'

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















