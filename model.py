import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model:int, vocab_size:int):       # d_model is the vectors' size / embeddings dimension
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)



class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, seq_len:int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len    # number of tokens
        self.dropout = nn.Dropout(dropout)    # to prevent overfitting


        # Matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)   # create a 1D vector like transpose(tensor([0., 1., 2., seq_len-1]))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))  # transformation of a vector tensor([0., 2., 4., seq_len-1]) for positional encoding
        # apply to even positions
        pe[:, 0::2] = torch.sin(position * div_term)  # We replace all the pair columns by column(position)*row(div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)   # (1, seq_len, d_model)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :  ]).requires_grad_(False)   # Add positional encodin to the vector
        return self.dropout(x)
    

class LayerNormalization(nn.Module):

    def __init__(self, eps:float):# epsilon : to avoid 0 in the denominator
        super().__init__()
        self.eps = eps 
        self.alpha = nn.Parameter(torch.ones(1))   # multiplied
        self.bias = nn.Parameter(torch.zeros(1))   # added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x-mean)/(std + self.eps) + self.bias
    





