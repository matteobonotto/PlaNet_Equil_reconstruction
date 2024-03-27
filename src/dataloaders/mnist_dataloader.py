import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from torch.utils.data import TensorDataset
import pandas as pd
from typing import Optional
# import matplotlib.pyplot as plt

DTYPE = torch.float32



class UserTrainDataset(Dataset):
    def __init__(self,path : str):
        super(UserTrainDataset,self).__init__()
        self.path = path

        # load and prepare data
        data = pd.read_csv(self.path, low_memory=False).to_numpy()
        X,y = data[:,1:], data[:,0]
        dims = X.shape
        dims = [int(x) for x in [dims[0], np.sqrt(dims[1]), np.sqrt(dims[1])]]
        X = X.reshape(dims)/255
        # plt.imshow(X[10000,:,:])
        # plt.show()
        self.X = torch.tensor(X,dtype=DTYPE).unsqueeze(1)
        self.y = torch.tensor(y,dtype=torch.int64)

    def __getitem__(self, idx):
        return self.X[idx, ...], self.y[idx, ...]
    
    def __len__(self):
        return self.y.shape[0]
    


class UserLightningDataModule(LightningDataModule):
    def __init__(
            self,
            path : str,
            batch_size : int = 32,
            shuffle : bool = False,
            num_workers : int = 0):
        super(UserLightningDataModule,self).__init__()
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = UserTrainDataset(path = self.path)
           
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        raise NotImplementedError
    
    def test_dataloader(self):
        raise NotImplementedError




class FashionMnistDataLoader(DataLoader):
    '''
    DEPRECATED
    '''
    def __init__(
            self,
            path : str,
            batch_size : int = 32,
            shuffle : bool = False,
            num_workers : int = 0):
        super(FashionMnistDataLoader,self).__init__(
            dataset = UserTrainDataset(path=path),
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers=num_workers
        )








