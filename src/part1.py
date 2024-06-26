import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

from transformer import *
from utilities import *

import numpy as np
import pandas as pd

from hyperparams import *

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
    #NOTE: this is NOT loading the vocab from the testing data despite the arg name
    print("Loading data and creating tokenizer ...")
    texts = load_texts(eval_data_dir)
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, os.path.join(eval_data_dir, "test_CLS.tsv"))
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    vocab_size = tokenizer.vocab_size
    return tokenizer, test_CLS_loader, vocab_size

def load_model_part1(vocab_size, device, model_path=None):
    print ("loading transformer encoder model...")
    te = CustomTransformerEncoder(device,  vocab_size, block_size, n_embd, n_head, n_layer, n_hidden, n_output).to(device)
    if model_path != None:
        te.load_state_dict(torch.load(model_path))
    return te

def train_part1(te, train_CLS_loader, device, data_dir):
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

        epoch_avg_accuracy = compute_classifier_accuracy(te, train_CLS_loader, device, data_dir)

        print(f"Epoch [{epoch+1}/{epochs_CLS}], Loss: {epoch_avg_loss:.4f}, Accuracy: {epoch_avg_accuracy:.4f}")
        losses.append(epoch_avg_loss)
        accs.append(epoch_avg_accuracy)
    training_metrics["loss"] = losses
    training_metrics["accuracy"] = accs
    training_metrics.to_csv(os.path.join(data_dir, "training_metrics", "transformer_encoder.csv"), sep=",", index=False)
    torch.save(te.state_dict(), os.path.join(data_dir, "models", "transformer_encoder.pt"))
    return losses, accs

def check_sanity_part1(te, tokenizer, device, plot_dir):
    print ("performing sanity check")
    u = Utilities(tokenizer, te, plot_dir, device)
    u.sanity_check("The quick brown fox jumped over the lazy dog.", block_size=block_size)
    u.sanity_check("Doing the same thing and expecting different results is insanity.", block_size=block_size)

def part1(device):
    data_dir = os.path.join(".", "data");
    training_data_dir = os.path.join("data", "speechesdataset")
    eval_data_dir = os.path.join("data", "speechesdataset")
    model_dir = os.path.join(data_dir, "models")
    plot_dir = os.path.join(data_dir, "plots", "part1")
    tokenizer, train_CLS_loader, vocab_size = load_training_data_part1(training_data_dir)
    te = load_model_part1(vocab_size, device, model_path=None)
    print ("loaded model")
    losses, accs = train_part1(te, train_CLS_loader, device, data_dir)
    print ("training done")
    print ("checking sanity")
    check_sanity_part1(te, tokenizer, device, plot_dir)
    _, test_CLS_loader, _ = load_eval_data_part1(eval_data_dir)
    eval_accuracy = compute_classifier_accuracy(te, test_CLS_loader, device, data_dir, write_data=True)
    print (f"eval accuracy: {eval_accuracy}")