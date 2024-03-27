import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import pandas as pd
import matplotlib.pyplot as plt

DTYPE = torch.float32


class FashionMnistDataloader():
    def __init__(self,path):
        self.path = path

    def prepare_data(self):
        data = pd.read_csv(self.path, low_memory=False).to_numpy()
        X,y = data[:,1:], data[:,0]
        dims = X.shape
        dims = [int(x) for x in [dims[0], np.sqrt(dims[1]), np.sqrt(dims[1])]]
        X = X.reshape(dims)/255
        # plt.imshow(X[10000,:,:])
        # plt.show()
        self.X = np.expand_dims(X,axis=1)
        self.y = y

    
    def dataloader(self,batch_size):
        self.prepare_data()
        return DataLoader(
            TensorDataset(
                torch.tensor(self.X,dtype=DTYPE),
                torch.tensor(self.y)),
            shuffle=True,
            batch_size=batch_size)




# class FashionMnistDataloader():
#     def __init__(self,path):
#         self.path = path

#         # load and prepare data
#         data = pd.read_csv(self.path, low_memory=False).to_numpy()
#         X,y = data[:,1:], data[:,0]
#         dims = X.shape
#         dims = [int(x) for x in [dims[0], np.sqrt(dims[1]), np.sqrt(dims[1])]]
#         X = X.reshape(dims)/255
#         # plt.imshow(X[10000,:,:])
#         # plt.show()
#         X = np.expand_dims(X,axis=1)
#         y = y

#         self.data = []
#         for X_i,y_i in zip()

#     def __getitem__(self, idx):
#         img_path, class_name = self.data[idx]

    
#     def dataloader(self,batch_size):
#         self.prepare_data()
#         return DataLoader(
#             TensorDataset(
#                 torch.tensor(self.X,dtype=DTYPE),
#                 torch.tensor(self.y)),
#             shuffle=True,
#             batch_size=batch_size)


