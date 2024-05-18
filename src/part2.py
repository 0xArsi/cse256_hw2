import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
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

    hbush_path = os.path.join(eval_data_dir, "test_LM_hbush.txt") 
    obama_path = os.path.join(eval_data_dir, "test_LM_obama.txt") 
    wbush_path = os.path.join(eval_data_dir, "test_LM_wbush.txt") 

    with open(hbush_path, 'r', encoding='utf-8') as f,open(obama_path, 'r', encoding='utf-8') as g,open(wbush_path, 'r', encoding='utf-8') as h:
        text_hbush = f.read()
        text_obama = g.read()
        text_wbush = h.read()


    test_hbush = LanguageModelingDataset(tokenizer, text_hbush, block_size)
    test_obama = LanguageModelingDataset(tokenizer, text_obama, block_size)
    test_wbush = LanguageModelingDataset(tokenizer, text_wbush, block_size)

    test_hbush_loader = DataLoader(test_hbush, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
    test_obama_loader = DataLoader(test_obama, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
    test_wbush_loader = DataLoader(test_wbush, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    vocab_size = tokenizer.vocab_size
    return tokenizer, test_hbush_loader, test_obama_loader, test_wbush_loader, vocab_size

def load_model_part2(vocab_size, device, model_path=None):
    print ("loading transformer decoder model...")
    td = CustomTransformerDecoder(device,  vocab_size, n_embd, n_head, n_layer, n_hidden).to(device)
    if model_path != None:
        td.load_state_dict(torch.load(model_path))
    return td

def train_part2(device, vocab_size, td, train_LM_loader, test_hbush_loader, test_obama_loader, test_wbush_loader, data_dir):
    criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
    optimizer = optim.Adam(td.parameters(), lr=0.001)  # Learning rate might need tuning
    training_metrics = pd.DataFrame({})
    testing_metrics = pd.DataFrame({})

    # for the classification  task, you will train for a fixed number of epochs like this:
    '''
    TRAIN MODEL
    '''
    losses = []
    perps = []
    perps_hbush = []
    perps_obama = []
    perps_wbush = []
    loss_over_eval_period = 0
    perp_over_eval_period = 0
    samples_over_eval_period = 0
    print ("starting training")

    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters: break
        td.train()  # Set the model to training mode
        xb, yb = xb.to(device), yb.to(device)  # Move data to the appropriate device
        # yb = F.one_hot(yb, num_classes=vocab_size)
        # print (f"x shape: {xb.shape}")
        # print (f"y shape: {yb.shape}")
        #zero out grads
        optimizer.zero_grad()
        #model prediction  
        outputs, _ = td(xb)  
        outputs_T = outputs.transpose(-2, -1)

        # print (f"output shape: {outputs.shape}")
        #use the last token in each sequence as the target 
        loss = criterion(outputs_T, yb)
        #backprop 
        loss.backward()
        optimizer.step()  

        loss_over_eval_period += loss.item()
        samples_over_eval_period += yb.size(0)

        if i % eval_interval == 0:
            perp_over_eval_period = compute_perplexity(td, train_LM_loader, device)
            perps.append(perp_over_eval_period)
            losses.append(loss_over_eval_period)
            print (f"[loss]: {loss_over_eval_period}, [train_perplexity]: {perp_over_eval_period}")
            loss_over_eval_period = 0
            samples_over_eval_period = 0

    perp_hbush = compute_perplexity(td, test_hbush_loader, device)
    perp_obama = compute_perplexity(td, test_obama_loader, device)
    perp_wbush = compute_perplexity(td, test_wbush_loader, device)

    perps_hbush.append(perp_hbush)
    perps_obama.append(perp_obama)
    perps_wbush.append(perp_wbush)
    perps.append(perp_over_eval_period)
    losses.append(loss_over_eval_period)
    loss_over_eval_period = 0
    samples_over_eval_period = 0

    training_metrics["period"] = np.arange(1, len(losses) + 1, dtype=int)
    training_metrics["loss"] = losses
    training_metrics["perplexity"] = perps
    
    testing_metrics["perplexity-hbush"] = perps_hbush
    testing_metrics["perplexity-obama"] = perps_obama
    testing_metrics["perplexity-wbush"] = perps_wbush

    training_metrics.to_csv(os.path.join(data_dir, "training_metrics", "transformer_decoder.csv"), sep=",", index=False)
    
    testing_metrics.to_csv(os.path.join(data_dir, "testing_metrics", "transformer_decoder.csv"), sep=",", index=False)

    torch.save(td.state_dict(), os.path.join(data_dir, "models", "transformer_decoder.pt"))
    return losses, perps


def check_sanity_part2(td, tokenizer, device, plot_dir):
    print ("performing sanity check")
    u = Utilities(tokenizer, td, plot_dir, device)
    u.sanity_check("The quick brown fox jumped over the lazy dog.", block_size=block_size)
    u.sanity_check("Doing the same thing and expecting different results is insanity.", block_size=block_size)

def part2(device):
    data_dir = os.path.join(".", "data");
    training_data_dir = os.path.join("data", "speechesdataset")
    eval_data_dir = os.path.join("data", "speechesdataset")
    model_dir = os.path.join(data_dir, "models")
    plot_dir = os.path.join(data_dir, "plots", "part2")
    tokenizer, train_LM_loader, vocab_size = load_training_data_part2(training_data_dir)
    _, test_hbush_loader, test_obama_loader, test_wbush_loader, _ = load_eval_data_part2(eval_data_dir)
    td = load_model_part2(vocab_size, device, model_path=None)
    losses, perps = train_part2(device, vocab_size, td, train_LM_loader, test_hbush_loader, test_obama_loader, test_wbush_loader, data_dir)
    check_sanity_part2(td, tokenizer, device, plot_dir)
    