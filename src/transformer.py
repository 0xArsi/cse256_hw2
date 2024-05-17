import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperparams import *

'''
layer implementations draw heavily from
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
but do not use any pre-made layers
'''
class PositionalEncoding(nn.Module):

    def __init__(self, device, max_len, n_embed, dropout=0.1):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1).to(self.device)
        div_term = torch.exp(torch.arange(0, n_embed, 2) * (-np.log(10000.0) / n_embed)).to(self.device)
        self.pe = torch.zeros(max_len, n_embed).to(self.device)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class AttentionHead(nn.Module):
    def __init__(self, n_embed, head_size, device, masked=False):
        super().__init__()
        self.n_embed = n_embed
        self.head_size = head_size
        self.masked = masked
        self.device = device
        self.query = torch.nn.Linear(n_embed, head_size)
        self.key = torch.nn.Linear(n_embed, head_size)
        self.value = torch.nn.Linear(n_embed, head_size)
        self.tril = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)).to(self.device)

    def forward(self, x):
        '''
        compute relevance of input embedding to the query vectors in learned 
        query matrix

        this result is multiplied by the key matrix to determine relevance
        of each query to each key

        multiply this result by the value matrix to obtain the relevance
        between input embedding the the value vectors (normalized)
        '''
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # matmul does not need contiguous inputs
        query_key_affinity_scores = torch.matmul(query, key.transpose(-2, -1)) * x.size(1)**-(1/2)
        if self.masked:
            query_key_affinity_scores = query_key_affinity_scores.masked_fill(self.tril[:block_size, :block_size], float("-inf"))
        attn = F.softmax(query_key_affinity_scores, dim=-1)
        ctx = torch.matmul(attn, value)

        # return attn maps too
        return ctx, attn

class NHeadAttention(nn.Module):
    def __init__(self, n_embed, n_heads, device, masked=False):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(n_embed, n_embed // n_heads, device, masked=masked) for _ in range (n_heads)])
    
    def forward(self, x):
        #concatenate attention head values along last dimension to get back n_embed
        ctx_tensors = []
        sample_attn_map = None
        for head in self.heads:
            ctx, attn = head(x)
            ctx_tensors.append(ctx)
            sample_attn_map = attn
        return torch.cat(ctx_tensors, dim=-1), sample_attn_map
    
'''
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers

## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training
'''
class Encoder(nn.Module):
    def __init__(self, device, n_embed, n_heads, n_hidden):
        super().__init__()
        #each token has an embedding of dimension n_embed. 
        #so the embedding for a sequence is a [batch_size, n_embed] matrix
        self.device = device
        self.layer_norm = nn.LayerNorm(n_embed)
        self.nhead_attn = NHeadAttention(n_embed, n_heads, device) 
        self.ff1 = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        x = self.layer_norm(x)
        ctx, sample_attn_map = self.nhead_attn(x)
        x = x + ctx
        x = x + self.ff1(x)
        return x, sample_attn_map

class Decoder(nn.Module):
    def __init__(self, device, vocab_size, n_embed, n_heads, n_hidden):
        super().__init__()
        #each token has an embedding of dimension n_embed. 
        #so the embedding for a sequence is a [batch_size, n_embed] matrix
        self.device = device
        self.layer_norm = nn.LayerNorm(n_embed)
        self.nhead_attn = NHeadAttention(n_embed, n_heads, device, masked=True) 
        self.ff1 = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        x = self.layer_norm(x)
        ctx, sample_attn_map = self.nhead_attn(x)
        x = x + ctx
        x = x + self.ff1(x)
        return x, sample_attn_map


class CustomTransformerEncoder(nn.Module):
    def __init__(self, device, vocab_size, max_len, n_embed, n_heads, n_layer, n_hidden, n_output):
        super().__init__()
        self.device = device 
        self.n_embed = n_embed
        #each token has an embedding of dimension n_embed. 
        #so the embedding for a sequence is a [batch_size, n_embed] matrix
        self.embedding = nn.Embedding(vocab_size, n_embed)
        #stack multihead attn layers followed by ff network
        self.positional_encoding = PositionalEncoding(self.device, max_len=max_len, n_embed=n_embed)
        self.encoder_layers = nn.ModuleList([Encoder(self.device, n_embed, n_heads, n_hidden) for _ in range(n_layer)])
        self.ffnet = nn.Sequential(
            nn.Linear(n_embed, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),

        )

    def forward(self, x):
        attention_maps = []
        # [batch_size, seq_len, n_embed]
        x = self.embedding(x).squeeze(dim=0)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x, sample_attn = layer(x)
            attention_maps.append(sample_attn)
        x = x.sum(dim=-2)
        x = self.ffnet(x)
        return x, attention_maps

class CustomTransformerDecoder(nn.Module):
    def __init__(self, device, vocab_size, n_embed, n_heads, n_layer, n_hidden):
        super().__init__()
        #each token has an embedding of dimension n_embed. 
        #so the embedding for a sequence is a [batch_size, n_embed] matrix
        self.device = device
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.embedding = nn.Embedding(vocab_size, n_embed)
        #stack multihead attn layers followed by ff network
        self.positional_encoding = PositionalEncoding(self.device, block_size, n_embed)
        self.decoder_layers = nn.ModuleList([Decoder(self.device, self.vocab_size, n_embed, n_heads, n_hidden) for _ in range(n_layer)])
        self.ffnet = nn.Sequential(
            nn.Linear(n_embed, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, vocab_size),
        )

    def forward(self, x):
        attention_maps = []
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.decoder_layers:
            x, attn = layer(x)
            attention_maps.append(attn)
        x = x.sum(dim=1)
        x = self.ffnet(x)
        return x, attention_maps