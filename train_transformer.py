#!/usr/bin/env python
import argparse
import gc
import os
import torch
from torch.utils.data import DataLoader
from data.sampler import BatchSampler
from data.Fbank import *
import torchaudio
from models.speech_transformer.decoder import Decoder
from models.speech_transformer.encoder import Encoder
from models.speech_transformer.transformer import Transformer
from solver.solver import Solver
from utils import *
from models.speech_transformer.optimizer import TransformerOptimizer

parser = argparse.ArgumentParser("Speech-Transformer Training ")

# General config
# Task related
parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp', help='location to download data')
parser.add_argument('--train_set', type=str, default='test-clean', help='train dataset')
parser.add_argument('--test_set', type=str, default='test-clean', help='test dataset')
parser.add_argument('--dict', type=str, default='spm/librispeech/1000_bpe.vocab',
                    help='Dictionary which should include <unk> <sos> <eos>')

# Network architecture
# encoder
# TODO: automatically infer input dim
parser.add_argument('--d_input', default=80, type=int,
                    help='Dim of encoder input')
parser.add_argument('--n_layers_enc', default=6, type=int,
                    help='Number of encoder stacks')
parser.add_argument('--n_head', default=8, type=int,
                    help='Number of Multi Head Attention (MHA)')
parser.add_argument('--d_k', default=64, type=int,
                    help='Dimension of key')
parser.add_argument('--d_v', default=64, type=int,
                    help='Dimension of value')
parser.add_argument('--d_model', default=512, type=int,
                    help='Dimension of model')
parser.add_argument('--d_inner', default=2048, type=int,
                    help='Dimension of inner')
parser.add_argument('--dropout', default=0.1, type=float,
                    help='Dropout rate')
parser.add_argument('--pe_maxlen', default=5000, type=int,
                    help='Positional Encoding max len')
# decoder
parser.add_argument('--d_word_vec', default=512, type=int,
                    help='Dim of decoder embedding')
parser.add_argument('--n_layers_dec', default=6, type=int,
                    help='Number of decoder stacks')
parser.add_argument('--tgt_emb_prj_weight_sharing', default=1, type=int,
                    help='share decoder embedding with decoder projection')
# Loss
parser.add_argument('--label_smoothing', default=0.1, type=float,
                    help='label smoothing')

# Training config
parser.add_argument('--epochs', default=30, type=int,
                    help='Number of maximum epochs')
# minibatch
parser.add_argument('--shuffle', default=0, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch-size', default=16, type=int,
                    help='Batch size')
parser.add_argument('--batch_frames', default=0, type=int,
                    help='Batch frames. If this is not 0, batch size will make no sense')
parser.add_argument('--maxlen-in', default=800, type=int, metavar='ML',
                    help='Batch size is reduced if the input sequence length > ML')
parser.add_argument('--maxlen-out', default=150, type=int, metavar='ML',
                    help='Batch size is reduced if the output sequence length > ML')
parser.add_argument('--num-workers', default=2, type=int,
                    help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--k', default=1.0, type=float,
                    help='tunable scalar multiply to learning rate')
parser.add_argument('--warmup_steps', default=4000, type=int,
                    help='warmup steps')
# save and load model
parser.add_argument('--save-folder', default='ckpts/speech_transformer',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue-from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model-path', default='final.pth.tar',
                    help='Location to save best validation model')
# logging
parser.add_argument('--print-freq', default=10, type=int,
                    help='Frequency of printing training infomation')
parser.add_argument('--visdom', dest='visdom', type=int, default=0,
                    help='Turn on visdom graphing')
parser.add_argument('--visdom_lr', dest='visdom_lr', type=int, default=0,
                    help='Turn on visdom graphing learning rate')
parser.add_argument('--visdom_epoch', dest='visdom_epoch', type=int, default=0,
                    help='Turn on visdom graphing each epoch')
parser.add_argument('--visdom-id', default='Transformer training',
                    help='Identifier for visdom run')


def main(args):

    # Load Data
    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)
    train_data = torchaudio.datasets.LIBRISPEECH(root=args.data_dir, url=args.train_set, download=True)
    test_data = torchaudio.datasets.LIBRISPEECH(root=args.data_dir, url=args.test_set, download=True)

    print('Sorting data for smart batching...')
    sorted_train_idxs = [idx for idx, _ in sorted(enumerate(train_data), key=lambda x: x[1][0].shape[1])]
    sorted_test_idxs = [idx for idx, _ in sorted(enumerate(test_data), key=lambda x: x[1][0].shape[1])]
    print('train data loading...')
    tr_loader = DataLoader(dataset=train_data,
                              pin_memory=True,
                              num_workers=args.num_workers,
                              batch_sampler=BatchSampler(sorted_train_idxs, batch_size=args.batch_size),
                              collate_fn=lambda x: preprocess_feature(x, 'train'))
    print('test data loading...')
    cv_loader = DataLoader(dataset=test_data,
                             pin_memory=True,
                             num_workers=args.num_workers,
                             batch_sampler=BatchSampler(sorted_test_idxs, batch_size=args.batch_size),
                             collate_fn=lambda x: preprocess_feature(x, 'valid'))
    print('data loader finished.')
    gc.collect()

    # load dictionary and generate char_list, sos_id, eos_id
    char_list, sos_id, eos_id = process_dict(args.dict)
    vocab_size = len(char_list)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    # model
    encoder = Encoder(args.d_input, args.n_layers_enc, args.n_head,
                      args.d_k, args.d_v, args.d_model, args.d_inner,
                      dropout=args.dropout, pe_maxlen=args.pe_maxlen)
    decoder = Decoder(sos_id, eos_id, vocab_size,
                      args.d_word_vec, args.n_layers_dec, args.n_head,
                      args.d_k, args.d_v, args.d_model, args.d_inner,
                      dropout=args.dropout,
                      tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                      pe_maxlen=args.pe_maxlen)
    
    # Print models size
    model_size(encoder, 'Encoder')
    model_size(decoder, 'Decoder')

    model = Transformer(encoder, decoder)
    print(model)
    model.cuda()
    # optimizer
    optimizier = TransformerOptimizer(
        torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        args.k,
        args.d_model,
        args.warmup_steps)

    # solver
    solver = Solver(data, model, optimizier, args)
    solver.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)