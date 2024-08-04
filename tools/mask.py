import torch

def make_pad_mask(lengths):
    """
    Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    """
    batch_size = int(lengths.size(0))
    max_len = int(lengths.max().item())
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

def pad_list(xs, pad_value):
    """
       Pad a list of tensors to the same length for batching.

       Parameters:
       - xs (list[torch.Tensor]): List of tensors to be padded.
       - pad_value (float or int): The value used for padding.

       Returns:
       - torch.Tensor: A padded tensor with shape (n_batch, max_len) + xs[0].shape[1:].

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad

def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
    """
    Generates a mask indicating which positions are non-filled

    Parameters:
    - padded_input (torch.Tensor): Tensor that contain padding elements.
    - input_lengths (list or torch.Tensor, optional): List of sequence lengths for each example in the batch.
    - pad_idx (int, optional): The padding value index in the tensor.

    Returns:
    - torch.Tensor: A mask tensor with the same batch dimension as `padded_input`,
      with an additional single dimension appended at the end. Positions corresponding to
      non-padded elements are set to True, while padded positions are set to False.
    """
    assert input_lengths is not None or pad_idx is not None
    if input_lengths is not None:
        # padded_input: N x T x ..
        N = padded_input.size(0)
        non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])  # N x T
        for i in range(N):
            non_pad_mask[i, input_lengths[i]:] = 0
    if pad_idx is not None:
        # padded_input: N x T
        assert padded_input.dim() == 2
        non_pad_mask = padded_input.ne(pad_idx).float()
    # unsqueeze(-1) for broadcast
    return non_pad_mask.unsqueeze(-1)

def get_subsequent_mask(seq):
    """
    Generates a mask for preventing a sequence from seeing its future positions.

    Parameters:
    - seq (torch.Tensor): (batch_size, sequence_length).

    Returns:
    - torch.Tensor: (batch_size, sequence_length, sequence_length),
                     where the upper triangle (including the diagonal) contains 1s and the rest contains 0s.
    """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)

    return subsequent_mask

def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
    """
    Generate a padding mask to ignore padded parts of the key sequence.

    Parameters:
    - seq_k (torch.Tensor): (batch_size, key_len).
    - seq_q (torch.Tensor): (batch_size, query_len).
    - pad_idx (int): The index value that represents padding in the sequence.

    Returns:
    - torch.Tensor: A boolean tensor of shape (batch_size, query_len, key_len), where positions
                    corresponding to the padding part of the key sequence are set to True, and others are set to False.
    """
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_attn_pad_mask(padded_input, input_lengths, expand_length):
    """
    Generate an attention mask.

    Parameters:
    - padded_input (torch.Tensor): Tensor containing padded input data.
    - input_lengths (list or torch.Tensor): Lengths of the input sequences without padding.
    - expand_length (int): Typically the maximum sequence length in the batch.

    Returns:
    - torch.Tensor: (batch_size, expand_length, sequence_length),
                    where positions corresponding to padded inputs are set to 0 (or masked),
                    and others are set to 1 (or unmasked).
    """
    non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
    pad_mask = non_pad_mask.squeeze(-1).lt(1)
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_mask