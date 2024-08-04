import torch
import torch.nn as nn
import numpy as np
from tools.common import remove_duplicates_and_blank

class GreedyCharacterDecoder(nn.Module):
    ''' Greedy CTC decoder - Argmax logits and remove duplicates. '''
    def __init__(self):
        super(GreedyCharacterDecoder, self).__init__()

    def forward(self, x):
        indices = torch.argmax(x, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = indices.tolist()
        hyp = remove_duplicates_and_blank(indices)
        return hyp


class BeamCharacterDecoder(nn.Module):
    def __init__(self, blank_index, eos_index, beam_width=3):
        super(BeamCharacterDecoder, self).__init__()
        self.blank_index = blank_index
        self.eos_index = eos_index
        self.beam_width = beam_width

    def forward(self, logits, seq_len):
        num_classes = logits.size(1)
        batch_size = logits.size(0)
        beam_width = self.beam_width
        num_steps = logits.size(0) // batch_size

        beams = [[] for _ in range(batch_size)]
        for b in range(batch_size):
            beams[b].append((0, [self.blank_index] * num_steps, 0))  # (概率, 序列, 累计概率)

        for step in range(num_steps):
            logits_step = logits[step * batch_size:(step + 1) * batch_size]
            logits_step = logits_step.cpu().numpy()

            new_beams = [[] for _ in range(batch_size)]
            backpointers = np.zeros((batch_size, beam_width), dtype=int)

            for b in range(batch_size):
                beam_item = beams[b]
                beam_scores = np.array([x[0] for x in beam_item])
                beam_scores = beam_scores.reshape(-1, 1)

                logits_step_expanded = np.repeat(logits_step, beam_width, axis=0)
                logits_step_expanded += beam_scores
                next_scores = np.apply_along_axis(lambda x: np.exp(x - np.max(x)), 1, logits_step_expanded)
                next_scores /= np.sum(next_scores, axis=1, keepdims=True)  # norm

                # choose top-k
                next_scores_flat = next_scores.flatten()
                k_indices = np.argsort(next_scores_flat)[::-1][:beam_width]
                next_scores_sorted = next_scores_flat[k_indices]
                next_indices = k_indices // num_classes
                char_indices = k_indices % num_classes

                for idx in range(beam_width):
                    backpointers[b, idx] = next_indices[idx]
                    new_seq = beams[b][next_indices[idx]][1] + [char_indices[idx]]
                    if char_indices[idx] == self.eos_index:
                        new_seq = new_seq[:-1]
                    new_beams[b].append((next_scores_sorted[idx], new_seq, beams[b][next_indices[idx]][2] + np.log(next_scores_sorted[idx])))

            for b in range(batch_size):
                new_beams[b] = [x for x in new_beams[b] if x[1][-1] != self.eos_index or len(x[1]) == 1]
                beams[b] = sorted(new_beams[b], key=lambda x: x[2], reverse=True)[:beam_width]

        for b in range(batch_size):
            beams[b].sort(key=lambda x: x[0], reverse=True)  # 按概率排序
            beams[b] = [x[1] for x in beams[b]]  # 只保留序列
        return beams
