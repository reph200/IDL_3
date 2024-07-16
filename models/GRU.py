import torch as tr
import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld
import matplotlib.pyplot as plt

from parameters import batch_size


# Implements GRU Unit

class ExGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()
        self.hidden_size = hidden_size
        self.sigmoid = torch.sigmoid

        # GRU Cell weights
        # Linear layers for reset gate, update gate, and candidate hidden state
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.out_gate = nn.Linear(input_size + hidden_size, hidden_size)

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)

    def name(self):
        return "GRU"

    def forward(self, x, hidden_state):
        # Implementation of GRU cell

        # Concatenate input and hidden state
        combined = torch.cat((x, hidden_state), 1)

        # Compute gates
        reset = self.sigmoid(self.reset_gate(combined))
        update = self.sigmoid(self.update_gate(combined))

        # Compute candidate hidden state
        combined_reset = torch.cat((x, reset * hidden_state), dim=1)
        h_candidate = torch.tanh(self.out_gate(combined_reset))

        # Update hidden state
        hidden_state = update * hidden_state + (1 - update) * h_candidate

        # Apply fully connected layer to the final hidden state
        output = self.fc(hidden_state)

        return output, hidden_state

    def init_hidden(self, bs=batch_size):
        return torch.zeros(bs, self.hidden_size)

