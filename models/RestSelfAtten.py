import torch as tr
import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
from models.MatMul import MatMul
from parameters import atten_size


class ExRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRestSelfAtten, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.atten_size = atten_size  # Fixed attention size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

        self.layer1 = MatMul(input_size, hidden_size)
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_k = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_v = MatMul(hidden_size, hidden_size, use_bias=False)
        self.output_layer = MatMul(hidden_size, output_size)

    def create_positional_encoding(self, seq_len, hidden_size):
        pos_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / hidden_size) for j in range(seq_len)]
            for pos in range(seq_len)
        ])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        return torch.tensor(pos_enc, dtype=torch.float32)

    def name(self):
        return "MLP_atten"

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        pos_enc = self.create_positional_encoding(seq_len, self.hidden_size).unsqueeze(0).repeat(batch_size, 1, 1)
        x = x + pos_enc

        x = self.layer1(x)
        x = self.ReLU(x)

        # Pad input
        padded = pad(x, (0, 0, self.atten_size, self.atten_size, 0, 0))

        x_nei = []
        for k in range(-self.atten_size, self.atten_size + 1):
            x_nei.append(torch.roll(padded, shifts=k, dims=1))

        x_nei = torch.stack(x_nei, dim=2)
        x_nei = x_nei[:, self.atten_size:-self.atten_size, :, :]

        queries = self.W_q(x).unsqueeze(2)
        keys = self.W_k(x_nei)
        values = self.W_v(x_nei)

        # scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.sqrt_hidden_size
        # atten_weights = self.softmax(scores)
        #
        # context = torch.matmul(atten_weights, values).sum(dim=2)
        # output = self.output_layer(context)


        scores = torch.einsum('bijd,bijd->bij', queries, keys) / self.sqrt_hidden_size

        # Apply softmax to get attention weights
        atten_weights = self.softmax(scores)

        # Weighted sum of values
        context = torch.einsum('bij,bijd->bid', atten_weights, values)/ self.sqrt_hidden_size
        output = self.output_layer(context)
        return output, atten_weights




# class ExLRestSelfAtten(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size):
#         super(ExRestSelfAtten, self).__init__()
#
#         self.input_size = input_size
#         self.output_size = output_size
#         self.sqrt_hidden_size = np.sqrt(float(hidden_size))
#         self.ReLU = torch.nn.ReLU()
#         self.softmax = torch.nn.Softmax(2)
#
#         # Token-wise MLP + Restricted Attention network implementation
#
#         self.layer1 = MatMul(input_size,hidden_size)
#         self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
#         # rest ...
#
#
#     def name(self):
#         return "MLP_atten"
#
#     def forward(self, x):
#
#         # Token-wise MLP + Restricted Attention network implementation
#
#         x = self.layer1(x)
#         x = self.ReLU(x)
#
#         # generating x in offsets between -atten_size and atten_size
#         # with zero padding at the ends
#
#         padded = pad(x,(0,0,atten_size,atten_size,0,0))
#
#         x_nei = []
#         for k in range(-atten_size,atten_size+1):
#             x_nei.append(torch.roll(padded, k, 1))
#
#         x_nei = torch.stack(x_nei,2)
#         x_nei = x_nei[:,atten_size:-atten_size,:]
#
#         # x_nei has an additional axis that corresponds to the offset
#
#         # Applying attention layer
#
#         # query = ...
#         # keys = ...
#         # vals = ...
#
#
#         return x, atten_weights