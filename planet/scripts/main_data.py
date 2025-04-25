
import random
import torch
from torch.utils.data import Dataset, DataLoader

random.seed(42)

class EquilDataset():
    def __init__(self):
        pass

    def __getitem__(self, idx: int):
        if random.random() > .5:
            # keep the full grid
            pass
        else:
            # interpolate on a subgrid
            pass




if __name__ == "__main__":
    # create the dataset with the equilibria on the full grid

















