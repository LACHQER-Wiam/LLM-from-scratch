import torch
import torch.nn as nn
import math



#### Embeddings & Positional encoding --------------------------------------------------
# Static predefined embeddings
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

        pe = pe.unsqueeze(0)   # the new dimension of pe is (1, seq_len, d_model)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x is the initial embedding
        """
        x = x + (self.pe[:, :x.shape[1], :  ]).requires_grad_(False)   # Add positional encoding to the vector
        return self.dropout(x)
    


#### Inside the encoder ----------------------------------------------------------------
class LayerNormalization(nn.Module):

    def __init__(self, eps:float = 10**-6):           # epsilon : to avoid 0 in the denominator
        super().__init__()
        self.eps = eps 
        self.alpha = nn.Parameter(torch.ones(1))   # multiplied
        self.bias = nn.Parameter(torch.zeros(1))   # added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True) 
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x-mean)/(std + self.eps) + self.bias
    

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff:int, dropout:float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)    # dot product + bias
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_ff) -> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model:int, h:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model 
        self.h = h     # number of attention's heads
        assert d_model % h == 0 , "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv

        self.w_o = nn.Linear(d_model, d_model)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) -> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None :
            attention_scores.masked_fill_(mask == 0, -1e9)     # Replace 0 per -1e9
        attention_scores = attention_scores.softmax(dim=-1)    # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)


        return (attention_scores @ value), attention_scores



    def forward(self, q, k, v, mask):
        query = self.w_q(q)     # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        key = self.w_k(k)     # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        value = self.w_v(v)     # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        
        # (Batch, seq_len, d_model) -> (Batch, seq_len, h, d_k) -> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k)  -> (Batch, seq_len, h, d_k) -> (Batch, seq_len, d_model) 
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        #  (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        return self.w_o(x)





    


    





