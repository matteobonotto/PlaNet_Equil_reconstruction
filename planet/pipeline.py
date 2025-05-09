from __future__ import annotations
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from numpy import ndarray

from .model import PlaNetCore


class PlaNet:
    def __init__(self, model: PlaNetCore, scaler: StandardScaler):
        self.model: PlaNetCore = model
        self.model.eval()
        self.scaler: StandardScaler = scaler

    def __call__(self, inputs) -> ndarray:
        pass

    @classmethod
    def from_pretrained(cls, path: str) -> PlaNet:
        pass
