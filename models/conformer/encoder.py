from torch import nn
from models.conformer.embedding import Conv2dSubsampling, RelPositionalEncoder
from models.conformer.convention import *
from models.conformer.attention import RelativeMultiHeadAttention
from models.conformer.feed_forword import FeedForwardBlock

class ConformerBlock(nn.Module):
    '''
    Conformer Encoder Block. 

    Parameters:
      d_model (int): Dimension of the models
      conv_kernel_size (int): Size of kernel to use for depthwise convolution
      feed_forward_residual_factor (float): output_weight for feed-forward residual connections
      feed_forward_expansion_factor (int): Expansion factor for feed-forward block
      num_heads (int): Number of heads to use for multi-head attention.py
      positional_encoder (nn.Module): PositionalEncoder module
      dropout (float): Dropout probability
    
    Inputs:
      x (Tensor): (batch_size, time, d_model)
      mask (Tensor): (batch_size, time, time) Optional mask to zero out attention.py score at certain indices
    
    Outputs:
      Tensor (batch_size, time, d_model): Output tensor from the conformer block.
    '''
    def __init__(
        self,
        d_model=144,
        conv_kernel_size=31,
        feed_forward_residual_factor=.5,
        feed_forward_expansion_factor=4,
        num_heads=4,
        positional_encoder=RelPositionalEncoder(144),
        dropout=0.1,
    ):
        super(ConformerBlock, self).__init__()
        self.residual_factor = feed_forward_residual_factor
        self.ff1 = FeedForwardBlock(d_model, feed_forward_expansion_factor, dropout)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout, positional_encoder)
        self.conv_block = ConvBlock(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForwardBlock(d_model, feed_forward_expansion_factor, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=6.1e-5)

    def forward(self, x, mask=None):
        x = x + (self.residual_factor * self.ff1(x))
        x = x + self.attention(x, mask=mask)
        x = x + self.conv_block(x)
        x = x + (self.residual_factor * self.ff2(x))
        return self.layer_norm(x)

class ConformerEncoder(nn.Module):
    '''
    Conformer Encoder Module. 

    Parameters:
      d_input (int): Dimension of the input
      d_model (int): Dimension of the models
      num_layers (int): Number of conformer blocks to use in the encoder
      conv_kernel_size (int): Size of kernel to use for depthwise convolution
      feed_forward_residual_factor (float): output_weight for feed-forward residual connections
      feed_forward_expansion_factor (int): Expansion factor for feed-forward block
      num_heads (int): Number of heads to use for multi-head attention.py
      dropout (float): Dropout probability
    
    Inputs:
      x (Tensor): input spectrogram of dimension (batch_size, time, d_input)
      mask (Tensor): (batch_size, time, time) Optional mask to zero out attention.py score at certain indices
    
    Outputs:
      Tensor (batch_size, time, d_model): Output tensor from the conformer encoder
      Tensor (batch_size, time, time): mask
    '''
    def __init__(
        self,
        d_input=80,
        d_model=144,
        num_layers=16,
        conv_kernel_size=31, 
        feed_forward_residual_factor=.5,
        feed_forward_expansion_factor=4,
        num_heads=4,
        dropout=0.1,
    ):
        super(ConformerEncoder, self).__init__()
        self.conv_subsample = Conv2dSubsampling(d_model=d_model)
        self.linear_proj = nn.Linear(d_model * (((d_input - 1) // 2 - 1) // 2), d_model) # project subsamples to d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # define global positional encoder to limit models parameters
        positional_encoder = RelPositionalEncoder(d_model)
        self.layers = nn.ModuleList([ConformerBlock(
            d_model=d_model,
            conv_kernel_size=conv_kernel_size, 
            feed_forward_residual_factor=feed_forward_residual_factor,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            num_heads=num_heads,
            positional_encoder=positional_encoder,
            dropout=dropout,
        ) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.conv_subsample(x)
        if mask is not None:
          mask = mask[:, :-2:2, :-2:2] # account for subsampling
          mask = mask[:, :-2:2, :-2:2] # account for subsampling
          assert mask.shape[1] == x.shape[1], f'{mask.shape} {x.shape}'
        
        x = self.linear_proj(x)
        x = self.dropout(x)
        
        for layer in self.layers:
          x = layer(x, mask=mask)
        
        return x
