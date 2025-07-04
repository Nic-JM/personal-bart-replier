import pandas as pd
import numpy as np
import random
from transformers import BartTokenizer, BartForConditionalGeneration
from encoded_data import EncodedData
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from model_selector import ModelSelector
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def determine_max_length(data):
    sample_matrix = random.sample(data, 3)

    #The data is a 2d array containing message and response
    sample_lengths = []
    for row in sample_matrix:
        #Pick either message or reply from the sampled 2d array
        sample_lengths.append(
            len(row[random.randint(0, 1)].split())
            )
    
    sample_mean = np.mean(sample_lengths)
    sample_std = np.std(sample_lengths)
    upper_limit = sample_mean + 3*sample_std #chebyshev/Normal cutoff

    # Since the number of tokens for a sentence are on average more than the number of words
    return int(upper_limit * 1.2)

def classify_message_length(message):
    message = str(message)
    length = len(message.split())

    if length < 3:
        return 0
    elif length >=3 and length < 9:
        return 1
    else:
        return 2

def tokenize_data(tokenizer, input, output, max_length):
    input_enc = tokenizer(input, max_length = max_length,
                        padding ="max_length",
                        truncation = True,
                        return_tensors = "pt")
    
    output_enc = tokenizer(output, max_length = max_length,
                        padding ="max_length",
                        truncation = True,
                        return_tensors = "pt")

    return input_enc, output_enc

def preprocess_data(data,  tokenizer):
    max_length = determine_max_length(data)
    enc_data = []

    for input, output in data:
        input = str(input)
        output = str(output)
        input_enc, output_enc = tokenize_data(tokenizer, input, output, max_length)
        labels = output_enc["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100 #pytorch crossentrpy ignores

        enc_data.append({
            "input_ids": input_enc["input_ids"][0],         # shape: [seq_len]
            "attention_mask": input_enc["attention_mask"][0],
            "labels": labels[0]
        })

    return enc_data
    
def print_loss_graph(train_loss_history, test_loss_history):

    plt_axis = range(1, len(train_loss_history) + 1)

    min_train_loss_epoch = np.argmin(train_loss_history) + 1
    min_train_loss = min(train_loss_history)

    min_test_loss_epoch = np.argmin(test_loss_history) + 1
    min_test_loss = min(test_loss_history)

    plt.plot(plt_axis, train_loss_history, label = "Training Loss", color = "blue")
    plt.plot(plt_axis, test_loss_history, label = "Test Loss", color = "orange")

    plt.scatter(min_train_loss_epoch, min_train_loss, color = "black", marker = 'o')
    plt.scatter(min_test_loss_epoch, min_test_loss, color = "black", marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Test and Train Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig("Test_Train_loss_graph.pdf", format="pdf")
    plt.close()

def main():
    loss_history = []
    test_loss_history = []
    loss_threshold = 0.0 # I want the lowest test loss
    models = ModelSelector(loss_threshold)

    #read in data
    df = pd.read_csv("message_reply_pairs.csv")
    data = df.values.tolist()

    X = [row[1] for row in data]
    y = [row[0] for row in data]
    strat_condition = [classify_message_length(row[1]) for row in data]

    # Train 90% | Test 10%
    train_input, train_output, test_input, test_output = train_test_split(X, y, 
                                                                          test_size=0.1, 
                                                                          random_state=42,
                                                                          stratify=strat_condition)


    train_dataset = list(zip(train_input, train_output))
    test_dataset = list(zip(test_input, test_output))

    # convert the input and output array into tokenized tensors
    tokeniser = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    processed_train_data = preprocess_data(train_dataset, tokeniser)
    processed_test_data = preprocess_data(test_dataset, tokeniser)

    ## load training data
    train_dataset = EncodedData(processed_train_data)
    training_data = DataLoader(train_dataset, batch_size=8, shuffle=True)

    ## Load test data
    test_dataset = EncodedData(processed_test_data)
    testing_data = DataLoader(test_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=(5e-5))
    lowest_loss = float("inf")
    lowest_test_loss = float("inf")
    epoch_without_improvement = 0
    epoch_without_test_improvement = 0

    for epoch in range(100):
        model.train()
        total_training_loss = 0

        # Train the model
        for batch in training_data:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_training_loss += loss.item()

        avg_loss = total_training_loss / len(training_data)
        loss_history.append(avg_loss)

        ### Test the models results ###
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in testing_data:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                output = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
                loss = output.loss
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(testing_data)
        test_loss_history.append(avg_test_loss)



        # Store the models
        if avg_test_loss < lowest_test_loss:
            lowest_test_loss = avg_test_loss
            epoch_without_test_improvement = 0
        else:
            epoch_without_test_improvement += 1

        if epoch_without_test_improvement <= 5:
            # Only save models that are improving on previous results
            models.add(epoch + 1, model.state_dict(), avg_test_loss)

        # Print loss update
        print(f"Epoch {epoch+1} | Avg Training Loss: {avg_loss:.4f} | Avg Test Loss: {avg_test_loss:.4f}")

        
        if avg_loss < lowest_loss:
            lowest_loss = avg_loss
            epoch_without_improvement = 0
        else:
            epoch_without_improvement += 1
        
        if epoch_without_improvement > 5:
            break


    best_epoch, best_model_state, best_test_loss = models.return_best_model()
    print_loss_graph(loss_history, test_loss_history)

    print(f"Saved the model from {best_epoch} with a test loss {best_test_loss:.4f}")
    torch.save(best_model_state, "best_bart_weights.pt")


if __name__== '__main__':
    main()
