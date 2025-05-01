from __future__ import annotations
from sklearn.preprocessing import StandardScaler

from .model import PlaNetCore


class PlaNet:
    def __init__(self, model: PlaNetCore, scaler: StandardScaler):
        self.model: PlaNetCore = model
        self.scaler: StandardScaler = scaler

    @classmethod
    def from_pretrained(cls, path: str) -> PlaNet:
        pass
