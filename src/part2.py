import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

from transformer import *
from utilities import Utilities

import numpy as np
import pandas as pd
from utilities import load_texts

from utilities import *
from hyperparams import *

def load_training_data_part2(training_data_dir):
    '''
    LOAD DATA
    '''
    print("Loading data and creating tokenizer ...")
    texts = load_texts(training_data_dir)
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)
    print ("loading data...")
    inputfile = os.path.join(training_data_dir, "train_LM.txt")
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    vocab_size = tokenizer.vocab_size
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
    return tokenizer, train_LM_loader, vocab_size



def load_eval_data_part2(eval_data_dir):
    #NOTE: this is NOT loading the vocab from the testing data despite the arg name
    print("Loading data and creating tokenizer ...")
    texts = load_texts(eval_data_dir)
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, os.path.join(eval_data_dir, "test_CLS.tsv"))
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    vocab_size = tokenizer.vocab_size
    return tokenizer, test_CLS_loader, vocab_size

def load_model_part2(vocab_size, device, model_path=None):
    print ("loading transformer decoder model...")
    te = CustomTransformerDecoder(device,  vocab_size, n_embd, n_head, n_layer, n_hidden).to(device)
    if model_path != None:
        te.load_state_dict(torch.load(model_path))
    return te

def train_part2(td, train_LM_loader, device, data_dir):
    criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
    optimizer = optim.Adam(td.parameters(), lr=0.001)  # Learning rate might need tuning
    training_metrics = pd.DataFrame({"epoch":np.arange(1, max_iters // eval_interval, dtype=int),"loss":np.zeros(max_iters, dtype=float), "accuracy": np.zeros(shape=max_iters, dtype=float)})

    # for the classification  task, you will train for a fixed number of epochs like this:
    '''
    TRAIN MODEL
    '''
    losses = []
    accs = []
    loss_over_eval_period = 0
    samples_over_eval_period = 0
    print ("starting training")

    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters: break
        td.train()  # Set the model to training mode
        total_loss = 0
        xb, yb = xb.to(device), yb.to(device)  # Move data to the appropriate device
        #zero out grads
        optimizer.zero_grad()
        #model prediction  
        outputs, _ = td(xb)  
        #compute loss
        loss = criterion(outputs, yb)
        #backprop 
        loss.backward()
        optimizer.step()  

        #update loss for this epoch
        total_loss += loss.item()  
        loss_over_eval_period += loss.item()
        samples_over_eval_period += yb.size(0)
            # break

        if i % eval_interval == 0:
            losses.append(loss_over_eval_period)
            accs.append(loss_over_eval_period * 100 / samples_over_eval_period)
            loss_over_eval_period = 0
            samples_over_eval_period = 0

        losses.append(epoch_avg_loss)
        accs.append(epoch_avg_accuracy)
    training_metrics["loss"] = losses
    training_metrics["accuracy"] = accs
    training_metrics.to_csv(os.path.join(data_dir, "training_metrics", "transformer_encoder.csv"), sep=",", index=False)
    torch.save(td.state_dict(), os.path.join(data_dir, "models", "transformer_encoder.pt"))
    return losses, accs

def train_part2():
    pass
    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    # for i, (xb, yb) in enumerate(train_LM_loader):
    #     if i >= max_iters:
    #         break
    #     xb, yb = xb.to(device), yb.to(device)
        # LM training code here

def part2(device):
    data_dir = os.path.join(".", "data");
    training_data_dir = os.path.join("data", "speechesdataset")
    eval_data_dir = os.path.join("data", "speechesdataset")
    model_dir = os.path.join(data_dir, "models")
    tokenizer, test_CLS_loader, vocab_size = load_training_data_part2(training_data_dir)
    load_model_part2(vocab_size, device, model_path=None)