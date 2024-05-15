import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import matplotlib
import matplotlib.pyplot as plt

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

from transformer import *
from utilities import Utilities

import numpy as np
import pandas as pd


'''
RANDOM SEED
'''
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f"device: {device}")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader, data_dir, write_data=False):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, _ = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
    
    accuracy = (100 * total_correct / total_samples)
    classifier.train()
    return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def load_training_data_part1(training_data_dir):
    print("Loading data and creating tokenizer ...")
    texts = load_texts(training_data_dir)
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, os.path.join(training_data_dir, "train_CLS.tsv"))
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    vocab_size = tokenizer.vocab_size
    return tokenizer, train_CLS_loader, vocab_size

def load_eval_data_part1(eval_data_dir):
    print("Loading data and creating tokenizer ...")
    texts = load_texts(eval_data_dir)
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, os.path.join(eval_data_dir, "test_CLS.tsv"))
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    vocab_size = tokenizer.vocab_size
    return tokenizer, test_CLS_loader, vocab_size

def load_model_part1(vocab_size, model_path=None):
    te = CustomTransformerEncoder(device,  vocab_size, block_size, n_embd, n_head, n_layer, n_hidden, n_output).to(device)
    if model_path != None:
        te.load_state_dict(torch.load(model_path))
    return te

def train_part1(te, train_CLS_loader, data_dir):
    criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
    optimizer = optim.Adam(te.parameters(), lr=0.001)  # Learning rate might need tuning
    training_metrics = pd.DataFrame({"epoch":np.arange(1, epochs_CLS+1, dtype=int),"loss":np.zeros(epochs_CLS, dtype=float), "accuracy": np.zeros(shape=epochs_CLS, dtype=float)})

    # for the classification  task, you will train for a fixed number of epochs like this:
    '''
    TRAIN MODEL
    '''
    losses = []
    accs = []
    print ("starting training")
    for epoch in range(epochs_CLS):
        te.train()  # Set the model to training mode
        total_loss = 0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)  # Move data to the appropriate device
            #zero out grads
            optimizer.zero_grad()
            #model prediction  
            outputs, _ = te(xb)  
            #compute loss
            loss = criterion(outputs, yb)
            #backprop 
            loss.backward()
            optimizer.step()  

            #update loss for this epoch
            total_loss += loss.item()  
            # break

        #print avg loss over epoch
        epoch_avg_loss = total_loss / len(train_CLS_loader)

        epoch_avg_accuracy = compute_classifier_accuracy(te, train_CLS_loader, data_dir)

        print(f"Epoch [{epoch+1}/{epochs_CLS}], Loss: {epoch_avg_loss:.4f}, Accuracy: {epoch_avg_accuracy:.4f}")
        losses.append(epoch_avg_loss)
        accs.append(epoch_avg_accuracy)
    training_metrics["loss"] = losses
    training_metrics["accuracy"] = accs
    training_metrics.to_csv(os.path.join(data_dir, "training_metrics", "transformer_encoder.csv"), sep=",", index=False)
    torch.save(te.state_dict(), os.path.join(data_dir, "models", "transformer_encoder.pt"))
    return losses, accs

def check_sanity_part1(te, tokenizer, plot_dir):
    print ("performing sanity check")
    u = Utilities(tokenizer, te, plot_dir, device)
    u.sanity_check("The quick brown fox jumped over the lazy dog.", block_size=block_size)
    u.sanity_check("Doing the same thing and expecting different results is insanity.", block_size=block_size)

def part1():
    data_dir = os.path.join(".", "data");
    training_data_dir = os.path.join("data", "speechesdataset")
    eval_data_dir = os.path.join("data", "speechesdataset")
    model_dir = os.path.join(data_dir, "models")
    plot_dir = os.path.join(data_dir, "plots", "part1")
    tokenizer, train_CLS_loader, vocab_size = load_training_data_part1(training_data_dir)
    te = load_model_part1(vocab_size=vocab_size, model_path=None)
    print ("loaded model")
    losses, accs = train_part1(te, train_CLS_loader, data_dir)
    print ("training done")
    print ("checking sanity")
    check_sanity_part1(te, tokenizer, plot_dir)
    _, test_CLS_loader, _ = load_eval_data_part1(eval_data_dir)
    eval_accuracy = compute_classifier_accuracy(te, test_CLS_loader, data_dir, write_data=True)
    print (f"eval accuracy: {eval_accuracy}")

def load_data_part2(data_dir):
    '''
    LOAD DATA
    '''
    print("Loading data and creating tokenizer ...")
    texts = load_texts(data_dir)
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)
    print ("loading data...")
    inputfile = os.path.join(data_dir, "train_LM.txt")
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
    print ("done")


def train_part2():
    pass
    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    # for i, (xb, yb) in enumerate(train_LM_loader):
    #     if i >= max_iters:
    #         break
    #     xb, yb = xb.to(device), yb.to(device)
        # LM training code here
def part2():
    pass
def part3():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("part", type=str)

    parsed_args = parser.parse_args()

    if parsed_args.part == "part1":
        part1()

    elif parsed_args.part == "part2":
        part2()
    
    elif parsed_args.part == "part3":
        part3()
    
