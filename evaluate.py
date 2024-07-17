import torch

from loader import preprocess_review, device, my_test_texts, my_test_labels


def print_review(rev_text, avg_sub_score, true_label, predicted_label, sub_scores):
    print(f"Review: {' '.join(rev_text)}")
    print(f"Avg Sub-scores: {avg_sub_score}")
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {predicted_label}")
    print("Sub-scores for each word:")
    for word, score in zip(rev_text, sub_scores):
        print(f"  {word}: {score}")
    print()

# Evaluation function for the MLP model
def evaluate_mlp_model(mlp_model_path, test_texts, test_labels):
    print("MLP model")

    mlp_model = torch.load(mlp_model_path)
    mlp_model.eval()  # Set the model to evaluation mode

    correct = 0
    total = 0

    with torch.no_grad():
        for review_text, true_label in zip(test_texts, test_labels):
            # Preprocess the review
            review_tensor = preprocess_review(review_text).to(device)

            # Get the model's output
            sub_scores = mlp_model(review_tensor)
            output = torch.mean(sub_scores, 1)
            _, predicted = torch.max(output, 1)

            # Convert predicted and true labels to integers
            true_label_int = 1 if true_label == "neg" else 0
            predicted_int = predicted.item()

            # Update the correct and total counts
            total += 1
            correct += (predicted_int == true_label_int)

            # Print the results for each review
            avg_sub_score = torch.mean(sub_scores, dim=1).detach()
            print_review(review_text.split(), avg_sub_score, true_label, 'positive' if predicted_int == 0 else 'negative', sub_scores.squeeze().tolist())

    accuracy = correct / total
    # print(f"\nAccuracy: {accuracy:.2f}")

# Evaluation function for the MLP_ATTN model
def evaluate_mlp_atten_model(mlp_atten_model_path, test_texts, test_labels):
    print("MLP_ATTEN model")
    mlp_atten_model = torch.load(mlp_atten_model_path)
    mlp_atten_model.eval()  # Set the model to evaluation mode

    correct = 0
    total = 0

    with torch.no_grad():
        for review_text, true_label in zip(test_texts, test_labels):
            # Preprocess the review
            review_tensor = preprocess_review(review_text).to(device)

            # Get the model's output
            sub_scores, atten_weights = mlp_atten_model(review_tensor)
            output = torch.mean(sub_scores, 1)
            _, predicted = torch.max(output, 1)

            # Convert predicted and true labels to integers
            true_label_int = 1 if true_label == "neg" else 0
            predicted_int = predicted.item()

            # Update the correct and total counts
            total += 1
            correct += (predicted_int == true_label_int)

            # Print the results for each review
            avg_sub_score = torch.mean(sub_scores,dim=1).detach()
            print_review(review_text.split(), avg_sub_score, true_label, 'positive' if predicted_int == 0 else 'negative', sub_scores.squeeze().tolist())

    accuracy = correct / total
    # print(f"\nAccuracy: {accuracy:.2f}")
    print()

if __name__ == '__main__':
    evaluate_mlp_model("MLP.pth", my_test_texts, my_test_labels)
    evaluate_mlp_atten_model("MLP_atten.pth", my_test_texts, my_test_labels)




