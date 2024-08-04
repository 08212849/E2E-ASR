from torch import nn
import torch
from models.conformer.embedding import RelPositionalEncoder
import math
import torch.nn.functional as F

class RelativeMultiHeadAttention(nn.Module):
    '''
      Relative Multi-Head Self-Attention Module. 
      Method proposed in Transformer-XL paper: https://arxiv.org/abs/1901.02860

      Parameters:
        d_model (int): Dimension of the models
        num_heads (int): Number of heads to split inputs into
        dropout (float): Dropout probability
        positional_encoder (nn.Module): PositionalEncoder module
      
      Inputs:
        x (Tensor): (batch_size, time, d_model)
        mask (Tensor): (batch_size, time, time) Optional mask to zero out attention.py score at certain indices
      
      Outputs:
        Tensor (batch_size, time, d_model): Output tensor from the attention.py module.
    
    '''
    def __init__(self, d_model=144, num_heads=4, dropout=0.1, positional_encoder=RelPositionalEncoder(144)):
        super(RelativeMultiHeadAttention, self).__init__()

        #dimensions
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        # Linear projection weights
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_pos = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model)

        # Trainable bias parameters
        self.u = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u)
        torch.nn.init.xavier_uniform_(self.v)

        self.layer_norm = nn.LayerNorm(d_model, eps=6.1e-5)
        self.positional_encoder = positional_encoder
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()

        # layer norm and pos embeddings
        x = self.layer_norm(x)
        pos_emb = self.positional_encoder(seq_length)
        pos_emb = pos_emb.repeat(batch_size, 1, 1)

        # Linear projections, split into heads
        q = self.W_q(x).view(batch_size, seq_length, self.num_heads, self.d_head)
        k = self.W_k(x).view(batch_size, seq_length, self.num_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, num_heads, d_head, time)
        v = self.W_v(x).view(batch_size, seq_length, self.num_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, num_heads, d_head, time)
        pos_emb = self.W_pos(pos_emb).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, num_heads, d_head, time)

        # Compute attention.py scores with relative position embeddings
        AC = torch.matmul((q + self.u).transpose(1, 2), k)
        BD = torch.matmul((q + self.v).transpose(1, 2), pos_emb)
        BD = self.rel_shift(BD)
        attn = (AC + BD) / math.sqrt(self.d_model)

        # Mask before softmax with large negative number
        if mask is not None:
          mask = mask.unsqueeze(1)
          mask_value = -1e+30 if attn.dtype == torch.float32 else -1e+4
          attn.masked_fill_(mask, mask_value)

        # Softmax
        attn = F.softmax(attn, -1)

        # Construct outputs from values
        output = torch.matmul(attn, v.transpose(2, 3)).transpose(1, 2) # (batch_size, time, num_heads, d_head)
        output = output.contiguous().view(batch_size, -1, self.d_model) # (batch_size, time, d_model)

        # Output projections and dropout
        output = self.W_out(output)
        return self.dropout(output)


    def rel_shift(self, emb):
        '''
          Pad and shift form relative positional encodings. 
          Taken from Transformer-XL implementation: https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py 
        '''
        batch_size, num_heads, seq_length1, seq_length2 = emb.size()
        zeros = emb.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_emb = torch.cat([zeros, emb], dim=-1)
        padded_emb = padded_emb.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        shifted_emb = padded_emb[:, :, 1:].view_as(emb)
        return shifted_emb

class MultiHeadAttention(nn.Module):
    '''
    Multi-Head Attention layer.
    Method proposed in Transformer paper: https://arxiv.org/abs/1706.03762

    Parameters:
        d_model (int): Dimension of the models
        num_heads (int): Number of heads to split inputs into
        dropout (float): Dropout probability

    Inputs:
        x (Tensor): (batch_size, time, d_model)
        mask (Tensor): (batch_size, time, time) Optional mask to zero out attention.py score at certain indices

    Outputs:
        Tensor (batch_size, time, d_model): Output tensor from the attention.py module.
    '''

    def __init__(self, num_heads=4, d_model=144, dropout=0.1):
        super().__init__()
        # dimensions
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        # Linear projection weights
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=6.1e-5)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()

        # layer norm and pos embeddings
        x = self.layer_norm(x)

        # Linear projections, split into heads
        q = self.W_q(x).view(batch_size, seq_length, self.num_heads, self.d_head)                     # (batch_size, d_head, num_heads, time)
        k = self.W_k(x).view(batch_size, seq_length, self.num_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, num_heads, d_head, time)
        v = self.W_v(x).view(batch_size, seq_length, self.num_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, num_heads, d_head, time)

        scores = torch.matmul(q.transpose(-2, -1), k) / math.sqrt(self.d_head)

        # Mask before softmax with large negative number
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask_value = -1e+30 if scores.dtype == torch.float32 else -1e+4
            scores.masked_fill_(mask, mask_value)

        # Softmax
        scores = F.softmax(scores, -1)

        # Construct outputs from values
        output = torch.matmul(scores, v.transpose(2, 3)).transpose(1, 2)  # (batch_size, time, num_heads, d_head)
        output = output.contiguous().view(batch_size, -1, self.d_model)  # (batch_size, time, d_model)

        # Output projections and dropout
        output = self.W_out(output)
        return self.dropout(output)