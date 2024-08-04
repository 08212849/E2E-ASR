# Pytorch E2E ASR
Pytorch implementation of [conformer](https://doi.org/10.21437/interspeech.2020-3015) model and [speech-transformer](http://dx.doi.org/10.1109/icassp.2018.8462506) model on the LibriSpeech dataset.

## Data Process

### Fbank.py:声学特征处理

声学特征部分采用了80维的Fbank，窗长25ms，帧移10ms。

数据增强：用了SpecAugment方法，掩码参数设为F=27，同时最大时域掩码覆盖率为Ps = 0.05，表示最大时域掩码的长度是音频时长的Ps倍。这里为了化简使得时频掩码为固定值。

处理后的数据返回（每个batch）：

| feature        | type         | size                                      |
| -------------- | ------------ | ----------------------------------------- |
| spectrograms   | torch.Tensor | (batch_size, time, freq)                  |
| labels         | torch.Tensor | (batch_size, label_lengths)               |
| input_lengths  | list         | batch_size(subsampling of time dimension) |
| labels_lengths | list         | batch_size                                |
| references     | list         | batch_size                                |
| mask           | torch.Tensor | (batch_size, time, time)                  |

### tokenizer.py:文本特征

character/subword tokenizer

英文e2e语音识别系统的模型单元可以是char，也可以是BPE(byte-pair-encoding)。一般来说，BPE效果更好。使用[sentencepiece](https://github.com/google/sentencepiece)工具在librispeech训练数据上对BPE进行训练。

- `<blank>`表示 CTC 的空白符号。
- `<unk>`表示未知的标记，任何词汇表之外的标记都将被映射到其中。
- `<sos/eos>`表示基于注意力机制的编码器解码器训练的语音开始和语音结束符号，它们共享相同的id。



## 1 Conformer

### Usage

Train model from scratch:

```
python train_conformer.py --data_dir=./data --train_set=train-clean-100 --test_set=test_clean
```
Resume training from checkpoint

```
python train_conformer.py --load_checkpoint --checkpoint_path=model_best.pt
```
Train with mixed precision: 

```
python train_conformer.py --use_amp
```

[Smart batching](https://mccormickml.com/2020/07/29/smart-batching-tutorial/) is used by default but may need to be disabled for larger datasets. For valid train_set and test_set values, see torchaudio's [LibriSpeech dataset](https://pytorch.org/audio/stable/datasets.html). 

The model parameters default to the Conformer (M) configuration. 

### Model

#### embedding.py

**Subsampling**

降采样/降帧率：输入序列越长，即帧的个数越多，网络计算量就越大。而在语音识别中，一定时间范围内的语音信号是相似的，多个连续帧对应的是同一个发音。subsampling是为减少后续encoder网络的前向计算、CTC loss和AED计算cross attention时的开销、加快解码速度。

语音任务里有两种使用CNN的方式，一种是2D-Conv，一种是1D-Conv：

- 2D-Conv: 输入数据看作是深度(通道数）为1，高度为F（Fbank特征维度，idim），宽度为T（帧数）的一张图。
- 1D-Conv: 输入数据看作是深度(通道数）为F（Fbank特征维度)，高度为1，宽度为T（帧数）的一张图。

Conformer中采用2D-Conv来实现降采样。选择把帧率降低4倍的网络`Conv2dSubsampling4`。

> encoder mask是原始帧率下的记录batch各序列长度的mask，在计算attention以及ctc loss时均要使用，现在帧数降低了，mask也要跟着变化。

**Positional Eembedding**

位置编码的用途是给特征编码加上位置信息。

Transformer实现位置编码的具体方式非常多，一直有新的位置编码形式被提出，包括可学习的位置编码、相对位置编码、RoPE、AliBi等等，也有许多关于Transformer位置编码的特性的讨论，包括长程衰减等等。

> [让研究人员绞尽脑汁的Transformer位置编码](https://spaces.ac.cn/archives/8130)

在relative position attention模块中使用，通过正弦和余弦函数的不同频率，为序列中的每个位置生成一个唯一的位置编码。返回独立的pos_emb，是因为在relative position attention中，需要获取relative pos_emb的信息。在标准attention中该返回值不会被用到。



#### attention.py

实现Multi-Headed Self-Attention Module。

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202408041730935.png" alt="image-20240730112809995" style="zoom:67%;" />

- **Multi-Head Self Attention（MHSA）**

  采用多头自注意力（MHSA），同时集成了 Transformer-XL的相对正弦位置编码方案。相对位置编码允许自我注意模块在不同的输入长度上更好地泛化，得到的编码器对话语长度的方差更稳健。使用带有 dropout 的 pre-norm 残差单元 [21, 22]，这有助于训练和正则化更深层次的模型。



#### convolution.py

实现Conformer中的Convolution Module。

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202408041729268.png" alt="image-20240730112539241" style="zoom: 67%;" />

GLU（Gated Linear Unit activation），按维度分割通道为两组，一组通过线性激活函数，另一组通过sigmoid函数。

Swish activation，一种平滑的非单调激活函数。

逐点卷积（pointwise convolution），用于通道扩展。

深度卷积（depthwise convolution），用于增加特征维度。



#### feed_forword.py

实现Conformer中的feed forward module。

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202408041730937.png" alt="image-20240730115720912" style="zoom: 67%;" />

## 2 Speech-Transformer

### Usage

Train model from scratch:

```
python train_transformer.py --data_dir=./data --train_set=train-clean-100 --test_set=test_clean
```

Resume training from checkpoint

```
python train_transformer.py --load_checkpoint --checkpoint_path=model_best.pt
```

Train with mixed precision: 

```
python train_transformer.py --use_amp
```

### Model

#### attention.py

实现Multi-head attention。Multi-head attention计算 h 次 Scaled Dot-Product Attention，其中 h 表示头数。

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202408041730939.png" alt="image-20240804164920601" style="zoom: 67%;" />

#### Transformer.py

Decoder的主要结构和Encoder类似，也包括embedding、Pos Encoding和decoder layer三部分，只是decoder layer相比encoder layer而言多了一个和encoder输出之间的交叉注意力。

<img src="https://qinglan-1324038201.cos.ap-nanjing.myqcloud.com/images/202408041736332.png" alt="image-20240804173638840" style="zoom:67%;" />

### tools/mask.py

Transformer网络结构实现要比Conformer Encoder简单一些，但Transformer需要考虑三种mask机制，而Conformer网络中只有Encoder self-attention。Transformer的掩码机制主要包括三部分：

- encoder的self attention的长度mask
- decoder的self attention的causal mask：通过原始输入和一个合适的上三角矩阵相乘（或者逻辑与）来得到，使得模型在进行预测时只能关注过去和当前的token，并确保它仅基于每个时间步骤可用的信息进行预测。
- encoder和decoder的cross-attention的mask：为综合encoder和decoder中的padding。cross attention中的Q来自decoder，需要和encoder中的key-value sets求相关性矩阵，只需要考虑padding。因此最后所产生memory_mask下方mask取决于decoder的padding，右方mask取决于encoder的padding。

| Model                 | Loss              | Tokenizer | WER        |
| --------------------- | ----------------- | --------- | ---------- |
| Speech-Transformer    | Cross Entropy     | BPE       | 12.8%      |
| Conformer-LSTM        | CTC               | Char/BPE  | 62.57%     |
| Conformer-Transformer | CTC+Cross Entropy | Char      | **10.09%** |
