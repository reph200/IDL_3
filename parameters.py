batch_size = 32
output_size = 2

hidden_size = 64  # Option 1
# hidden_size = 128  # Option 2

run_recurrent = False  # else run Token-wise MLP
use_RNN = False  # otherwise GRU
atten_size = 0  # atten > 0 means using restricted self atten

reload_model = False
num_epochs = 10
learning_rate = 0.001
test_interval = 50

max_len_selected_review = 20
