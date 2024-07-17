import torch as tr
import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld
import matplotlib.pyplot as plt
from models.GRU import ExGRU
from models.MLP import ExMLP
from models.RNN import ExRNN
from models.RestSelfAtten import ExRestSelfAtten
from parameters import run_recurrent, use_RNN, output_size, hidden_size, atten_size, learning_rate, num_epochs, \
    test_interval, batch_size, reload_model, max_len_selected_review

# Loading dataset, use toy = True for obtaining a smaller dataset

train_dataset, test_dataset, num_words, input_size = ld.get_data_set(batch_size)

if __name__ == '__main__':

    if run_recurrent:
        if use_RNN:
            model = ExRNN(input_size, output_size, hidden_size)
        else:
            model = ExGRU(input_size, output_size, hidden_size)
    else:
        if atten_size > 0:
            model = ExRestSelfAtten(input_size, output_size, hidden_size)
        else:
            model = ExMLP(input_size, output_size, hidden_size)

    print("Using model: " + model.name())

    if reload_model:
        print("Reloading model")
        # model.load_state_dict(torch.load(model.name() + ".pth"))
        # model = torch.load('MLP_atten.pth')
        model = torch.load('MLP.pth')


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    steps = []


    def calculate_accuracy(outputs, labels):
        _, predicted = torch.max(outputs, 1)
        _, labels = torch.max(labels, 1)
        correct = (predicted == labels).sum().item()
        return correct / labels.size(0)


    train_loss = 1.0
    test_loss = 1.0

    # Training steps in which a test step is executed every test_interval

    for epoch in range(num_epochs):

        itr = 0  # iteration counter within each epoch

        for labels, reviews, reviews_text in train_dataset:  # getting training batches

            itr = itr + 1
            if (itr + 1) % test_interval == 0:
                test_iter = True
                labels, reviews, reviews_text = next(iter(test_dataset))  # get a test batch
            else:
                test_iter = False

            # Recurrent nets (RNN/GRU)

            if run_recurrent:
                hidden_state = model.init_hidden(int(labels.shape[0]))

                for i in range(num_words):
                    output, hidden_state = model(reviews[:, i, :], hidden_state)  # HIDE

            else:

                # Token-wise networks (MLP / MLP + atten)
                if atten_size > 0:
                    # MLP + atten
                    sub_score, atten_weights = model(reviews)
                else:
                    # MLP
                    sub_score = model(reviews)

                output = torch.mean(sub_score, 1)

            # cross-entropy loss
            loss = criterion(output, torch.argmax(labels, dim=1))

            # optimize in training iterations
            if not test_iter:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # averaged losses
            if test_iter:
                test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
                test_accuracy = calculate_accuracy(output, labels)
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)
                steps.append(itr + len(train_losses))  # track test steps
            else:
                train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss
                train_accuracy = calculate_accuracy(output, labels)
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)
                steps.append(itr + len(train_losses))  # track training steps

            if test_iter:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{itr + 1}/{len(train_dataset)}], "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Test Loss: {test_loss:.4f}, "
                    f"Train Accuracy: {train_accuracy:.4f}, "
                    f"Test Accuracy: {test_accuracy:.4f}"
                )

                if not run_recurrent:
                    # Assuming sub_score, labels, and output are tensors
                    nump_subs = sub_score.detach().numpy()  # Convert tensor to NumPy array
                    labels_np = labels.detach().numpy()  # Convert tensor to NumPy array

                    # Calculate average sub-scores across the batch
                    avg_sub_scores = np.mean(nump_subs, axis=1)

                    # Assuming output is a tensor, convert it to a NumPy array
                    output_np = output.detach().numpy()



                # saving the model
                torch.save(model, model.name() + ".pth")
