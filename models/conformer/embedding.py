from torch import nn
import torch
import math
import numpy as np

class PositionalEncoder(nn.Module):
    """
    Positional encoding.
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))

    Args:
        d_model(int): embedding dim
        dropout_rate(float): dropout rate
        int max_len(int): maximum input length
    """

    def __init__(self, d_model, dropout_rate, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        self.pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x, offset=0):
        """
        Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int): position offset

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """
        assert offset + x.size(1) < self.max_len
        self.pe = self.pe.to(x.device)
        pos_emb = self.pe[:, offset:offset + x.size(1)]
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset, size):
        """
        For getting encoding in a streaming fashion

        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int): start offset
            size (int): requried size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        assert offset + size < self.max_len
        return self.dropout(self.pe[:, offset:offset + size])

class RelPositionalEncoder(nn.Module):
    '''
      Generate positional encodings used in the relative multi-head attention.py module.
      These encodings are the same as the original transformer models: https://arxiv.org/abs/1706.03762

      Parameters:
        max_len (int): Maximum sequence length (time dimension)

      Inputs:
        len (int): Length of encodings to retrieve

      Outputs
        Tensor (len, d_model): Positional encodings
    '''

    def __init__(self, d_model, max_len=5000):
        super(RelPositionalEncoder, self).__init__()
        self.d_model = d_model
        encodings = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float)
        inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        encodings[:, 0::2] = torch.sin(pos[:, None] * inv_freq)
        encodings[:, 1::2] = torch.cos(pos[:, None] * inv_freq)
        self.register_buffer('encodings', encodings)

    def forward(self, len):
        return self.encodings[:len, :]

class Conv2dSubsampling(nn.Module):
    '''
    2d Convolutional subsampling. 
    Subsamples time and freq domains of input spectrograms by a factor of 4, d_model times. 

    Parameters:
      d_model (int): Dimension of the models
    
    Inputs:
      x (Tensor): Input spectrogram (batch_size, time, d_input)
    
    Outputs:
      Tensor (batch_size, time, d_model * (d_input // 4)): Output tensor from the conlutional subsampling module

    '''
    def __init__(self, d_model=144):
        super(Conv2dSubsampling, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=d_model, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, x):
        output = self.module(x.unsqueeze(1)) # (batch_size, 1, time, d_input)
        batch_size, d_model, subsampled_time, subsampled_freq = output.size()
        output = output.permute(0, 2, 1, 3)
        output = output.contiguous().view(batch_size, subsampled_time, d_model * subsampled_freq)
        return output

def pos_sinusoid_embedding(seq_len, d_model):
    embeddings = torch.zeros((seq_len, d_model))
    for i in range(d_model):
        f = torch.sin if i % 2 == 0 else torch.cos
        embeddings[:, i] = f(torch.arange(0, seq_len) / np.power(1e4, 2 * (i // 2) / d_model))
    return embeddings.float()