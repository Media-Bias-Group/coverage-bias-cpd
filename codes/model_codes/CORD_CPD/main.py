import time
import argparse
import pickle
import os
import datetime

from trainer import Trainer
from tester import Tester

def parse_args(cmd=None):
    parser = argparse.ArgumentParser()

    # experiment setting
    parser.add_argument('--mode', type=str, choices=['train', 'test']) #train or test
    parser.add_argument('--exp_dir', type=str, default='exp/exp') #experiment directory for saving results and models
    parser.add_argument('--no-cuda', action='store_true', help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--load', action='store_true', help='for fine tunning. Leave empty to train from scratch')

    # data setting
    parser.add_argument('--data_norm', type=str, default="mean_std") #data normalization method, "mean_std" or "min_max"
    parser.add_argument('--data_path', type=str, default="data/cp_change") #data path for training and testing
    parser.add_argument('--data_type', type=str, default="sim") #data type, "sim" for simulated data and "real" for real-world data. Note that the current code only supports simulated data, and real-world data will require additional processing and adjustments to the model and training procedure.

    # training setting
    parser.add_argument('--epochs', type=int, default=50000,   #training epochs
                        help='Number of epochs to train.')
    parser.add_argument('--batch-size', type=int, default=128, #batch size for training and testing
                        help='Number of samples per batch.')
    parser.add_argument('--lr', type=float, default=0.001, #learning rate
                        help='Initial learning rate.')
    parser.add_argument('--eval_epoch', type=int, default=5) #evaluate the model every eval_epoch epochs

    # model
    parser.add_argument('--spatial-encoding-layer', type=str, default="gnn", choices=["gnn", "transformer"], #spatial encoder
                        help='spatial encoder')
    parser.add_argument('--temporal-encoding-layer', type=str, default="transformer", choices=["rnn", "transformer"], #temporal encode
                        help='temporal encoder')
    parser.add_argument('--decoder', type=str, default='mlp') #decoder type
    parser.add_argument('--dims', type=int, default=4, #input dimension
                        help='The number of input dimensions (position + velocity).')
    parser.add_argument('--encoder-hidden', type=int, default=256, #hidden units for encoder
                        help='Number of hidden units.')
    parser.add_argument('--decoder-hidden', type=int, default=256, #hidden units for decoder
                        help='Number of hidden units.')
    parser.add_argument('--edge-types', type=int, default=2, #number of edge types
                        help='The number of edge types to infer.')
    parser.add_argument('--temp', type=float, default=0.5,
                        help='Temperature for Gumbel softmax.')
    parser.add_argument('--num-atoms', type=int, default=5, #num atoms in simulation
                        help='Number of atoms in simulation.')
    parser.add_argument('--no-factor', action='store_true', default=False,
                        help='Disables factor graph model.')
    parser.add_argument('--suffix', type=str, default='variable_5')
    parser.add_argument('--encoder-dropout', type=float, default=0.0, #dropout rate for encoder
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--decoder-dropout', type=float, default=0.0, #dropout rate for decoder
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--timesteps', type=int, default=100, #number of time steps per sample
                        help='The number of time steps per sample.')
    parser.add_argument('--prediction-steps', type=int, default=10, metavar='N', # number of steps to predict before re-using teacher forcing
                        help='Num steps to predict before re-using teacher forcing.')
    parser.add_argument('--begin-steps', type=int, default=0, # number of steps to skip at the beginning
                        help='Num steps begin to predict')
    parser.add_argument('--lr-decay', type=int, default=200, # after how many epochs to decay LR
                        help='After how epochs to decay LR by a factor of gamma.')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='LR decay factor.')
    parser.add_argument('--skip-first', action='store_true', default=True,
                        help='Skip first edge type in decoder, i.e. it represents no-edge.')
    parser.add_argument('--var', type=float, default=5e-5,
                        help='Output variance.')
    parser.add_argument('--hard', action='store_true', default=False,
                        help='Uses discrete samples in training forward pass.')
    parser.add_argument('--dynamic-graph', action='store_true', default=False,
                        help='Whether test with dynamically re-computed graph.')


    if cmd is None: # parse from command line
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd.split())

    print(args)

    # if args.dynamic_graph:
    #     print("Testing with dynamically re-computed graph.")
    return args

def main():
    args = parse_args()

    print("exp name: {}".format(args.exp_dir))
    print(f"mode: {args.mode}")
    if args.mode == 'train':
        train = Trainer(args)
        train.data_type = args.data_type
        train.report_combine = True # set to True if using real data
        train.logging("process data")
        train.load_data()
        train.logging("set model and train")
        train.set_model()
        train.train()
    elif args.mode == 'test':
        test = Tester(args)
        test.solve()
    else:
        raise RuntimeError(f'invalid mode')

if __name__ == '__main__':
    main()