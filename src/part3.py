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

from ray import train, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from hyperparams import *

def load_training_data_part3(training_data_dir):
    print("Loading data and creating tokenizer ...")
    texts = load_texts(training_data_dir)
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, os.path.join(training_data_dir, "train_CLS.tsv"))
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    vocab_size = tokenizer.vocab_size
    return tokenizer, train_CLS_loader, vocab_size

def load_eval_data_part3(eval_data_dir):
    #NOTE: this is NOT loading the vocab from the testing data despite the arg name
    print("Loading data and creating tokenizer ...")
    texts = load_texts(eval_data_dir)
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, os.path.join(eval_data_dir, "test_CLS.tsv"))
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    vocab_size = tokenizer.vocab_size
    return tokenizer, test_CLS_loader, vocab_size

def load_model_part3(vocab_size, device, model_path=None, p_dropout=0):
    print ("loading transformer encoder model...")
    te = CustomTransformerEncoder(device,  vocab_size, block_size, n_embd, n_head, n_layer, n_hidden, n_output, p_dropout=p_dropout).to(device)
    if model_path != None:
        te.load_state_dict(torch.load(model_path))
    return te

def train_part3(te, train_CLS_loader, device, data_dir, alpha=1e-3):
    criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
    optimizer = optim.Adam(te.parameters(), lr=alpha)  # Learning rate might need tuning
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
    training_metrics.to_csv(os.path.join(data_dir, "training_metrics", "transformer_encoder_enhanced.csv"), sep=",", index=False)
    torch.save(te.state_dict(), os.path.join(data_dir, "models", "transformer_encoder_enhanced.pt"))
    return losses, accs

def check_sanity_part3(te, tokenizer, device, plot_dir):
    print ("performing sanity check")
    u = Utilities(tokenizer, te, plot_dir, device)
    u.sanity_check("The quick brown fox jumped over the lazy dog.", block_size=block_size)
    u.sanity_check("Doing the same thing and expecting different results is insanity.", block_size=block_size)



def tune_part3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(".", "data");
    training_data_dir = os.path.join("data", "speechesdataset")
    eval_data_dir = os.path.join("data", "speechesdataset")
    model_dir = os.path.join(data_dir, "models")
    plot_dir = os.path.join(data_dir, "plots", "part3")
    tokenizer, train_CLS_loader, vocab_size = load_training_data_part3(training_data_dir)
    _, test_CLS_loader, _ = load_eval_data_part3(eval_data_dir)
    print ("tuning model...")
    param_space = {
    "n_embed": tune.choice([32, 64, 128, 256, 512]),
    "n_heads": tune.choice([2, 4, 8]),
    "n_layer": tune.choice([2, 4, 6]),
    "n_hidden": tune.choice([100, 128, 256]),
    "p_dropout": 0, 
    "batch_size": 16,
    "alpha": tune.loguniform(1e-4, 1e-2),
    }
    scheduler = ASHAScheduler(metric="loss", mode="min", max_t=9, grace_period=1, reduction_factor=2)
    prog_reporter = CLIReporter(metric_columns=["epoch_avg_loss", "epoch"])

    tuning_result = tune.run(
        tune.with_parameters(train_for_tuning, device=device, train_CLS_loader=train_CLS_loader, vocab_size=vocab_size), 
        resources_per_trial={"cpu": 4, "gpu": 1}, 
        config=param_space, 
        num_samples=20, 
        scheduler=scheduler, 
        progress_reporter=prog_reporter
    )
    best_te = tuning_result.get_best_trial("loss", "min", "last")
    print(f"Best te params: {best_te.config}")
    n_embed_best = best_te.config["n_embed"]
    n_head_best = best_te.config["n_heads"]
    n_layer_best = best_te.config["n_layer"]
    n_hidden_best = best_te.config["n_embed"]
    alpha_best = best_te.config["alpha"]
    cte_best = CustomTransformerEncoder(device,  vocab_size, block_size, n_embed_best, n_head_best, n_layer_best, n_hidden_best, n_output, p_dropout=0).to(device)
    train_part3(cte_best, train_CLS_loader, device, data_dir, alpha=alpha_best)
    eval_accuracy = compute_classifier_accuracy(cte_best, test_CLS_loader, device, data_dir, write_data=True)
    print(f"Best te params: {best_te.config}")
    print (f"eval_accuracy: {eval_accuracy}")
    print ("done")
    

def train_for_tuning(config, device=None, train_CLS_loader=None, vocab_size=5575, data_dir=None):
    te = CustomTransformerEncoder(device=device, vocab_size=vocab_size, max_len=block_size, n_embed=config["n_embed"], n_heads=config["n_heads"], n_layer=config["n_layer"], n_hidden=config["n_hidden"], n_output=n_output, p_dropout=config["p_dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
    optimizer = optim.Adam(te.parameters(), lr=config["alpha"])  # Learning rate might need tuning

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
        train.report({"loss": epoch_avg_loss})
    
def part3(device, p_dropout=0, tune_also=False):
    data_dir = os.path.join(".", "data");
    training_data_dir = os.path.join("data", "speechesdataset")
    eval_data_dir = os.path.join("data", "speechesdataset")
    model_dir = os.path.join(data_dir, "models")
    plot_dir = os.path.join(data_dir, "plots", "part3")
    tokenizer, train_CLS_loader, vocab_size = load_training_data_part3(training_data_dir)
    te = load_model_part3(vocab_size, device, model_path=None, p_dropout=p_dropout)
    print ("loaded model")
    losses, accs = train_part3(te, train_CLS_loader, device, data_dir)
    print ("training done")
    print ("checking sanity")
    check_sanity_part3(te, tokenizer, device, plot_dir)
    _, test_CLS_loader, _ = load_eval_data_part3(eval_data_dir)
    eval_accuracy = compute_classifier_accuracy(te, test_CLS_loader, device, data_dir, write_data=True)
    print (f"eval accuracy: {eval_accuracy}")
    print ("done")

