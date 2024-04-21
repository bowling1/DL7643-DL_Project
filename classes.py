# KA 04/13/23 -- Added classes to build transformer model. Please feel free to make necessary edits
import torch
import torch.nn as nn
import torch.optim as optim

import math
import numpy as np

# Citation -- https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
# Leverages the already-existing PyTorch implementation. Using their example as a base
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
#Citation -- https://medium.com/@eugenesh4work/attention-mechanism-for-lstm-used-in-a-sequence-to-sequence-task-be1d54919876
class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers):
        super(EncoderLSTM, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True)
        
    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: (batch_size, seq_len, hidden_dim)
        # decoder_hidden: (batch_size, hidden_dim)
        # Calculate the attention scores.
        scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)  # (batch_size, seq_len)
        
        attn_weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)
        
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_dim)

        return context_vector, attn_weights

class DecoderLSTMWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers):
        super(DecoderLSTMWithAttention, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.attention = Attention()

    def forward(self, input, encoder_outputs, hidden, cell):
        input = input.unsqueeze(1)  # (batch_size, 1)
        embedded = self.embedding(input)  # (batch_size, 1, emb_dim)
        
        context_vector, attn_weights = self.attention(encoder_outputs, hidden[-1])  # using the last layer's hidden state

        rnn_input = torch.cat([embedded, context_vector.unsqueeze(1)], dim=2)  # (batch_size, 1, emb_dim + hidden_dim)

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.out(output.squeeze(1))
        
        return prediction, hidden, cell
class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """
    # Constructor
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
        use_lstm = False
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
             dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)

    def forward(
        self,
        src,
        tgt,
    ):
        # Commenting these out as these are handled by Bokyoung's encode function

        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)
        # print(tgt.size())
        # print(src.size())
        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        #src = src.to(torch.long)

        # src = self.embedding(src) * math.sqrt(self.dim_model)
        # tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        # src = self.positional_encoder(src)
        # tgt = self.positional_encoder(tgt)

        # we permute to obtain size (sequence length, batch_size, dim_model),
        # src = src.permute(1, 0, 2)
        # tgt = tgt.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt)
        out = self.out(transformer_out)

        return out
    def encode(self, x):
        x = self.embedding(x) * math.sqrt(self.dim_model)
        x = self.positional_encoder(x)
        x = x.permute(1, 0, 2)
        return x

    def decode(self, x):
        x = self.embedding(x) * math.sqrt(self.dim_model)
        x = self.positional_encoder(x)
        x = x.permute(1, 0, 2)
        return x
    
    def lstm_integration(self, x):
        if self.use_lstm:
            out, _ = self.lstm(x)
            return out
        else:
            pass

    

