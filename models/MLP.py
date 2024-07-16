import torch as tr
import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld
import matplotlib.pyplot as plt

from models.MatMul import MatMul


class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP, self).__init__()

        self.ReLU = torch.nn.ReLU()

        # Token-wise MLP network weights
        self.layer1 = MatMul(input_size, hidden_size)
        self.layer2 = MatMul(hidden_size, output_size)

    def name(self):
        return "MLP"

    def forward(self, x):
        # Token-wise MLP network implementation
        x = self.layer1(x)
        x = self.ReLU(x)
        x = self.layer2(x)
        return x
