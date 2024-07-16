import torch as tr
import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld
import matplotlib.pyplot as plt


# Implements RNN Unit

class ExRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRNN, self).__init__()

        self.hidden_size = hidden_size
        self.sigmoid = torch.sigmoid

        # RNN Cell weights
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden2out = nn.Linear(hidden_size, output_size)

        # what else?

    def name(self):
        return "RNN"

    def forward(self, x, hidden_state):
        # Implementation of RNN cell
        outputs = []
        for t in range(x.size(1)):  # Iterate over time steps
            combined = torch.cat((x[t, :], hidden_state), dim=1)  # Concatenate input and hidden state
            hidden_state = torch.tanh(self.in2hidden(combined))  # Update hidden state
            output = self.hidden2out(hidden_state)  # Calculate output
            outputs.append(output.unsqueeze(1))  # Append output to list
        outputs = torch.cat(outputs, dim=1)
        return outputs, hidden_state

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)
