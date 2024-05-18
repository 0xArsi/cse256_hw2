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


import numpy as np
import pandas as pd

from part1 import part1
from part2 import part2
from part3 import part3
# from part3 import part3

from utilities import Utilities
from transformer import *
from hyperparams import *

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f"device: {device}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("part", type=str)

    parsed_args = parser.parse_args()

    if parsed_args.part == "part1":
        part1(device)

    elif parsed_args.part == "part2":
        part2(device)
    
    elif parsed_args.part == "part3":
        part3(device, p_dropout=0.4)
    
