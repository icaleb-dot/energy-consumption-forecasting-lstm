from abc import ABC, abstractmethod

import torch.nn as nn

class BaseNN(nn.Module, ABC):
    def __init__(self, name):
        super(BaseNN, self).__init__()
        self.name = name

    @abstractmethod
    def forward(self, x):
        pass