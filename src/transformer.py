import numpy as np
import torch
import torch.nn as nn

'''
@note
Computes embeddings for a batch of tokens.
The embedding of a token is the sum of its 
token embedding and positional embedding
'''
class CompositeEmbedding(nn.Module):
    def __init__(self, vocab_size, n_embed):
        super().__init__()
        #token embedding and positional embedding
        #we need one embedding for every word in the vocab
        self.token_embeddings = nn.Embedding(vocab_size, n_embed)
        self.positional_embeddings = nn.Embedding(vocab_size, n_embed)
        #normalize the sums of the embedding vectors
        self.normalize = nn.LayerNorm(n_embed, eps=1e-5)
    
    def forward(self, indices):
        # create input position tensor that accounts for different sequence lengths
        input_positions = torch.arange(indices.size(1), device=indices.device, dtype=torch.long).expand_as(indices)
        token_em = self.token_embeddings(indices)
        positional_em = self.positional_embeddings(input_positions)
        composite_em = token_em + positional_em
        normalized_em = self.normalize(composite_em)
        #consider adding dropout here if embedding dimension is large
        return normalized_em

class NHeadAttention(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        self.n_embed = n_embed
        self.num_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.query = torch.nn.Linear(n_embed, n_embed)
        self.key = torch.nn.Linear(n_embed, n_embed)
        self.value = torch.nn.Linear(n_embed, n_embed)
        self.normalize = nn.LayerNorm(n_embed, eps=1e-5)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        '''
        compute relevance of input embedding to the query vectors in 
        query matrix

        this result is multiplied by the key matrix to determine relevance
        of each query to each key

        multiply this result by the value matrix to obtain the relevance
        between input embedding the the value vectors (normalized)
        '''
        #get batch size from number of embedding matrices in x 
        #view query tensor grouped by head instead of across all heads
        batch_size = x.size(0)

        # compute q, k, v PER HEAD but in one tensor
        query = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        #value is of shape [batch_size, seq_length, num_heads, head_dim]
        # matmul does not need contiguous inputs
        raw_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, device=x.device))
        attn = torch.softmax(raw_scores, dim=-1)
        attn = self.dropout(attn)

        #view requires contiguous inputs
        #multiply by value mtrx to get weighted value vectors for each set of attn values per embd matrix, PER HEAD
        #reshape to [batch_size, embed_dim] to get the result per embedding (aggregate the heads)
        ctx = torch.matmul(attn, value).transpose(1, 2).contiguous().view(batch_size, -1, self.n_embed)

        # normalize attn context
        x = self.normalize(ctx)
        return x
    
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
class CustomTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_embed, n_heads, n_layer, n_input, n_hidden, n_output):
        super().__init__()
        #each token has an embedding of dimension n_embed. 
        #so the embedding for a sequence is a [batch_size, n_embed] matrix
        self.embedding = CompositeEmbedding(vocab_size, n_embed)
        #stack multihead attn layers followed by ff network
        self.transformer_layers = nn.ModuleList([NHeadAttention(n_embed, n_heads) for _ in range(n_layer)])
        self.ffnet = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.transformer_layers:
            x = layer(x)
        # take the mean over all the embeddings for each token
        # gives a length-indep. repr. for each sequence
        x = x.mean(dim=1)
        x = self.ffnet(x)
        return x