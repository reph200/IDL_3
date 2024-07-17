import torch

from parameters import run_recurrent, atten_size, max_len_selected_review
from sentiment_start import test_dataset, num_words, print_review, input_size


def evalute_MLP():
    model = torch.load('MLP.pth')
    # Evaluate the model on the test data and print examples of TP, TN, FN, FP
    true_positive = None
    true_negative = None
    false_positive = None
    false_negative = None

    for labels, reviews, reviews_text in test_dataset:
        if run_recurrent:
            hidden_state = model.init_hidden(int(labels.shape[0]))
            for i in range(num_words):
                output, hidden_state = model(reviews[:, i, :], hidden_state)
        else:
            if atten_size > 0:
                sub_score, atten_weights = model(reviews)
            else:
                sub_score = model(reviews)
            output = torch.mean(sub_score, 1)

        _, predicted = torch.max(output, 1)
        _, true_labels = torch.max(labels, 1)

        avg_sub_score = torch.mean(sub_score, dim=1).detach().numpy()

        for i in range(len(predicted)):
            if len(reviews_text[i]) < max_len_selected_review:  # Restrict reviews to under 15 words
                if predicted[i] == 1 and true_labels[i] == 1:
                    if true_positive is None:
                        true_positive = (reviews_text[i], avg_sub_score[i], true_labels[i].item(), predicted[i].item(),
                                         sub_score[i].detach().numpy())
                elif predicted[i] == 0 and true_labels[i] == 0:
                    if true_negative is None:
                        true_negative = (reviews_text[i], avg_sub_score[i], true_labels[i].item(), predicted[i].item(),
                                         sub_score[i].detach().numpy())
                elif predicted[i] == 1 and true_labels[i] == 0:
                    if false_positive is None:
                        false_positive = (reviews_text[i], avg_sub_score[i], true_labels[i].item(), predicted[i].item(),
                                          sub_score[i].detach().numpy())
                elif predicted[i] == 0 and true_labels[i] == 1:
                    if false_negative is None:
                        false_negative = (reviews_text[i], avg_sub_score[i], true_labels[i].item(), predicted[i].item(),
                                          sub_score[i].detach().numpy())

    print("\nTrue Positive:")
    if true_positive:
        print_review(*true_positive)
    else:
        print("No True Positive example found.")

    print("\nTrue Negative:")
    if true_negative:
        print_review(*true_negative)
    else:
        print("No True Negative example found.")

    print("\nFalse Positive:")
    if false_positive:
        print_review(*false_positive)
    else:
        print("No False Positive example found.")

    print("\nFalse Negative:")
    if false_negative:
        print_review(*false_negative)
    else:
        print("No False Negative example found.")

def evalute_MLP_atten():
    # model = torch.load('MLP_atten.pth')
    model = torch.load('MLP.pth')
    # Function to evaluate the model on given reviews
    def evaluate_reviews(reviews, true_labels):
        reviews_tensor = []  # Preprocess the reviews into tensors
        reviews_text = []  # Store the original review text
        for review in reviews:
            review_tensor, review_text = preprocess_review(review)
            reviews_tensor.append(review_tensor)
            reviews_text.append(review_text)

        reviews_tensor = torch.stack(reviews_tensor)
        true_labels_tensor = torch.tensor(true_labels, dtype=torch.long)

        if run_recurrent:
            hidden_state = model.init_hidden(len(reviews_tensor))
            for i in range(num_words):
                output, hidden_state = model(reviews_tensor[:, i, :], hidden_state)
        else:
            if atten_size > 0:
                sub_score, atten_weights = model(reviews_tensor)
            else:
                sub_score = model(reviews_tensor)
            output = torch.mean(sub_score, 1)

        _, predicted = torch.max(output, 1)

        for i in range(len(predicted)):
            avg_sub_score = torch.mean(sub_score[i], dim=0).detach().numpy()
            print_review(reviews_text[i], avg_sub_score, true_labels_tensor[i].item(), predicted[i].item(),
                         sub_score[i].detach().numpy())


    def preprocess_review(review):
        # Dummy preprocessing function, should be replaced with actual preprocessing
        # Here, it's assumed the review is already tokenized and converted to tensor
        tokens = review.split()
        review_tensor = torch.zeros(num_words, input_size)
        for i, token in enumerate(tokens[:num_words]):
            # Here you need to convert token to an index, for now, it is dummy values
            review_tensor[i] = torch.tensor([float(ord(c)) for c in token] + [0] * (input_size - len(token)))
        return review_tensor, tokens


    # List of 4 reviews to evaluate (dummy data, replace with actual reviews)
    reviews_to_evaluate = [
        "comment this movie is impossible is terrible very improbable bad interpretation direction not look.",
        "ming the merciless does little bardwork and movie most foul",
        "just love the interplay between two great characters of stage screen veidt barrymore",
        "this is the definitive movie version of hamlet branagh cuts nothing but there are no wasted moments",
        "it was really bad movie it was not like last time hope next time will be fun","very good good very good good bad"
    ]

    # List of true labels for the 4 reviews (dummy data, replace with actual labels)
    true_labels = [1, 1, 0, 0, 1,1]

    evaluate_reviews(reviews_to_evaluate, true_labels)

if __name__ == '__main__':
    # evalute_MLP()
    evalute_MLP_atten()