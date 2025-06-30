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
    
def print_loss_graph(loss_history, lowest_loss, loss_threshold):
    loss_cutoff = lowest_loss * (1 + loss_threshold)
    plt.axhline(y = loss_cutoff, color="red")
    plt.plot(range(1, len(loss_history) + 1),loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig("loss_graph.pdf", format="pdf") 

def main():
    loss_history = []
    loss_threshold = 0.5
    models = ModelSelector(loss_threshold)

    #test inputs
    inputs = ["Hi there!", "How are you?", "What's up?"]
    outputs = ["Hey!", "I'm fine, thanks.", "Not much, you?"]
    data = []
    for i in range(3):
        data.append([inputs[i], outputs[i]])

    #read in data
    df = pd.read_csv("message_reply_pairs.csv")
    data = df.values.tolist()
    
    # convert the input and output array into tokenized tensors
    tokeniser = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    processed_data = preprocess_data(data, tokeniser)
    dataset = EncodedData(processed_data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=(5e-5))
    lowest_loss = float("inf")
    epoch_without_improvement = 0

    for epoch in range(25):
        model.train()
        total_loss = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)

        # Store the models
        models.add(epoch + 1, model, avg_loss)

        # Print loss update
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        
        if avg_loss < lowest_loss:
            lowest_loss = avg_loss
            epoch_without_improvement = 0
        else:
            epoch_without_improvement += 1
        
        if epoch_without_improvement > 4:
            break


    best_epoch, best_model, best_loss = models.return_best_model()
    print_loss_graph(loss_history, lowest_loss, loss_threshold)
    print(f"The model found in {best_epoch[:-1]} {best_epoch[-1]} has the best loss of {best_loss}")
    torch.save(best_model, "best_bart_model.pt")


if __name__== '__main__':
    main()
