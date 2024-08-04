import torchaudio
import torch
import torch.nn as nn
from data.tokenizer import TextTransform

def get_audio_specaugment():
  '''
    Build an audio conversion pipeline for training and validation.
    Using SpecAugment for data augmentation.
    
    Returns:
        train_audio_transform (torch.nn.Sequential)
        valid_audio_transform (torchaudio.transforms.MelSpectrogram)
    
    Details:
      For simplicity, we approximate the size of time mask with just a fixed value.
    - freq_mask_param: 27
    - time_mask_param: 15
    - p: 0.05

  '''
  time_masks = [torchaudio.transforms.TimeMasking(time_mask_param=15, p=0.05) for _ in range(10)]
  train_audio_transform = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160), 
    torchaudio.transforms.FrequencyMasking(freq_mask_param=27),
    *time_masks,
  )
  valid_audio_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160)
  return train_audio_transform, valid_audio_transform


def preprocess_feature(data, data_type="train"):
  '''
    Perform preprocessing on LibriSpeech batch data.
    
    Parameters:
      - batch_data: data list.
      - data_type: indicates training or validation data.

     Returns:
        A tuple containing:
            - spectrograms (torch.Tensor)
            - labels (torch.Tensor)
            - input_lengths (list)
            - label_lengths (list)
            - references (list)
            - mask (torch.Tensor)
  '''
  text_transform = TextTransform()
  train_audio_transform, valid_audio_transform = get_audio_specaugment()
  spectrograms = []
  labels = []
  references = []
  input_lengths = []
  label_lengths = []

  for (waveform, _, utterance, _, _, _) in data:
    # Spectrogram
    if data_type == 'train':
      spec = train_audio_transform(waveform).squeeze(0).transpose(0, 1) # (1, time, freq)
    else:
      spec = valid_audio_transform(waveform).squeeze(0).transpose(0, 1) # (1, time, freq)
    spectrograms.append(spec)
    # Labels 
    references.append(utterance)
    label = torch.Tensor(text_transform.text_to_int(utterance))
    labels.append(label)
    # Lengths (time)
    input_lengths.append(((spec.shape[0] - 1) // 2 - 1) // 2) # account for subsampling of time dimension
    label_lengths.append(len(label))

  # Padding
  spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
  labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

  # Padding mask (batch_size, time, time)
  mask = torch.ones(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[1])
  for i, l in enumerate(input_lengths):
    mask[i, :, :l] = 0

  return spectrograms, labels, input_lengths, label_lengths, references, mask.bool()


if __name__ == '__main__':
  # test get_audio_specaugment()
  train_audio_transform, valid_audio_transform = get_audio_specaugment()
  print(train_audio_transform)
  print(valid_audio_transform)
