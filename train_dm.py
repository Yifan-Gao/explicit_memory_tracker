import torch
import random
import numpy as np
from argparse import ArgumentParser
from dm.base_entail import Module

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_batch', default=10, type=int, help='training batch size')
    parser.add_argument('--dev_batch', default=5, type=int, help='dev batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--epoch', default=5, type=int, help='number of epochs')
    parser.add_argument('--keep', default=1, type=int, help='number of model saves to keep')
    parser.add_argument('--seed', default=3, type=int, help='random seed')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0.35, type=float, help='dropout rate')
    parser.add_argument('--warmup', default=0.1, type=float, help='optimizer warmup')
    parser.add_argument('--loss_span_weight', default=1., type=float, help='span loss weight')
    parser.add_argument('--span_top_k', default=20, type=int, help='extract top k spans')
    parser.add_argument('--loss_entail_weight', default=1., type=float, help='entailment loss weight')
    parser.add_argument('--debug', action='store_true', help='debug flag to load less data')
    parser.add_argument('--dsave', default='save/{}', help='save directory')
    parser.add_argument('--model', default='c2f_entail', help='model to use')
    parser.add_argument('--early_stop', default='dev_combined', help='early stopping metric')
    parser.add_argument('--bert_hidden_size', default=768, type=int, help='hidden size for the bert model')
    parser.add_argument('--data', default='data', help='directory for data')
    parser.add_argument('--data_type', default='entail_bu', help='data type, b:base bert, l: large bert, u: uncased, c: cased')
    parser.add_argument('--prefix', default='default', help='prefix for experiment name')
    parser.add_argument('--bert_model_path', default='', help='bert model path')
    parser.add_argument('--eval_every_steps', default=200, type=int, help='evaluate model every xxx steps')

    args = parser.parse_args()
    args.dsave = args.dsave.format(args.prefix)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device('cuda')

    if args.debug:
        limit = 27
        data = {k: torch.load('{}/proc_{}_{}.pt'.format(args.data, args.data_type, 'dev'))[:limit] for k in ['train', 'dev']}
    else:
        limit = None
        data = {k: torch.load('{}/proc_{}_{}.pt'.format(args.data, args.data_type, k))[:limit] for k in ['dev', 'train']}

    print('instanting model')
    model = Module.load_module(args.model)(args, device)

    model.to(model.device)

    model.run_train(data['train'], data['dev'])
