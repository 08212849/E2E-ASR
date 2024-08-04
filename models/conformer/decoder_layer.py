from torch import nn
import torch

class TransformerDecoderLayer(nn.Module):
    """
    Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention.py module instance.
        src_attn (torch.nn.Module): Self-attention.py module instance.
        feed_forward (torch.nn.Module): Feed-forward module instance.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention.py layer's inpu
            and output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)

    Returns:
        torch.Tensor: Output tensor (#batch, maxlen_out, size).
        torch.Tensor: Mask for output tensor (#batch, maxlen_out).
        torch.Tensor: Encoded memory (#batch, maxlen_in, size).
        torch.Tensor: Encoded memory mask (#batch, maxlen_in).
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout_rate, normalize_before=True, concat_after=False):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-12)
        self.norm2 = nn.LayerNorm(size, eps=1e-12)
        self.norm3 = nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.concat_linear1 = nn.Linear(size + size, size)
        self.concat_linear2 = nn.Linear(size + size, size)

    def forward(self, tgt, tgt_mask, memory, memory_mask, cache=None):
        """
        Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor
                (batch, maxlen_out).
            memory (torch.Tensor): Encoded memory
                (batch, maxlen_in, size).
            memory_mask (torch.Tensor): Encoded memory mask
                (batch, maxlen_in).
            cache (torch.Tensor): cached tensors.
                (batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor (batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (batch, maxlen_out).
            torch.Tensor: Encoded memory (batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (batch, maxlen_in).

        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), "{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = tgt_mask[:, -1:, :]

        if self.concat_after:
            tgt_concat = torch.cat((tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1)
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = torch.cat((x, self.src_attn(x, memory, memory, memory_mask)), dim=-1)
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask
