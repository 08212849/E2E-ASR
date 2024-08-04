from models.conformer.feed_forword import *
from models.conformer.attention import *
from models.conformer.embedding import *
from models.conformer.decoder_layer import *
from tools.mask import *

class LSTMDecoder(nn.Module):
    '''
      LSTM Decoder

      Parameters:
        d_encoder (int): Output dimension of the encoder
        d_decoder (int): Hidden dimension of the decoder
        num_layers (int): Number of LSTM layers to use in the decoder
        num_classes (int): Number of output classes to predict
      
      Inputs:
        x (Tensor): (batch_size, time, d_encoder)
      
      Outputs:
        Tensor (batch_size, time, num_classes): Class prediction logits
    
    '''
    def __init__(self, d_encoder=256, d_decoder=640, num_layers=1, num_classes=29):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size=d_encoder, hidden_size=d_decoder, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(d_decoder, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        logits = self.linear(x)
        return logits

class TransformerDecoder(torch.nn.Module):
    """Base class of Transfomer decoder module.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention.py
        attention_heads: the number of heads of multi head attention.py
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention.py
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after: whether to concat attention.py layer's input and output
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        super().__init__()
        attention_dim = encoder_output_size

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                PositionalEncoder(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(
                f"only 'embed' is supported: {input_layer}")

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-12)
        self.use_output_layer = use_output_layer
        self.output_layer = torch.nn.Linear(attention_dim, vocab_size)

        self.decoders = torch.nn.ModuleList([
            TransformerDecoderLayer(
                attention_dim,
                MultiHeadAttention(attention_dim, attention_heads,  self_attention_dropout_rate),
                MultiHeadAttention(attention_dim, attention_heads,  src_attention_dropout_rate),
                FeedForwardBlock(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(num_blocks)
        ])

    def forward(self, memory, memory_mask, ys_in_pad, ys_in_lens):
        """
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out, vocab_size)
                    if use_output_layer is True,
                olens: (batch, )
        """
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens).unsqueeze(1)).to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        x, _ = self.embed(tgt)
        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory, memory_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.use_output_layer:
            x = self.output_layer(x)

        olens = tgt_mask.sum(1)
        return x, olens

    def forward_one_step(self, memory, memory_mask, tgt, tgt_mask, cache=None):
        """
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x, _ = self.embed(tgt)
        new_cache = []
        for i, decoder in enumerate(self.decoders):
            if cache is None:
                c = None
            else:
                c = cache[i]
            x, tgt_mask, memory, memory_mask = decoder(x, tgt_mask, memory, memory_mask, cache=c)
            new_cache.append(x)
        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.use_output_layer:
            y = torch.log_softmax(self.output_layer(y), dim=-1)
        return y, new_cache
