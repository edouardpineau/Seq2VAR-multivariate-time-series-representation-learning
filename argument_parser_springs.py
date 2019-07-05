import argparse
import torch


def argument_parser():
    """ Creation of argument parser for the experiments """

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--batch-size', type=int, default=64, help='Number of samples per batch.')
    parser.add_argument('--in-dim', type=int, default=1, help='Number of dimension per atom of the MTS.')
    parser.add_argument('--lag', action='store_true', default=2, help='Lag in the Granger causality model.')
    parser.add_argument('--num-atoms', type=int, default=10, help='Number of atoms in simulation.')
    parser.add_argument('--nb-systems', type=int, default=10, help='Number of different dynamical systems.')
    parser.add_argument('--nb-samples-per-system', type=int, default=100, help='Number of samples per dynamical system.')
    parser.add_argument('--prior', type=int, default=0.5, help='Sparsity prior.')
    parser.add_argument('--sd', type=int, default=0, help='Standard deviation of the Gaussian additive noise.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--stationary', action='store_true', default=True, help='Each time series is stationary.')
    parser.add_argument('--suffix', type=str, default='_springs10', help='Suffix for training data (e.g. "_charged".')
    parser.add_argument('--timesteps', type=int, default=25, help='Length of MTS samples.')
    
    args, unknown = parser.parse_known_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    
    args.path = '/data/tsi/analyse_de_donnees/04-Data/Springs/Datasets/same_graph/'
    
    return args


def argument_parser_seq2var():
    """ Creation of argument parser for the Seq2VAR experiments """

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder-hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--lr', action='store_true', default=5e-4, help='Learning rate.')
    parser.add_argument('--tau', type=int, default=0.5, help='Gumbel softmax temperature.')
    
    args, unknown = parser.parse_known_args()
    
    return args


def argument_parser_nri():
    """ Creation of argument parser for the NRI (https://arxiv.org/abs/1802.04687) experiments.
    
    Stolen from https://github.com/ethanfetaya/NRI
    """

    parser_nri = argparse.ArgumentParser()
    parser_nri.add_argument('--cuda', action='store_true', default=True, help='Enables CUDA training.')

    parser_nri.add_argument('--dynamic-graph', action='store_true', default=False, help='Whether test with dynamically re-computed graph.')
    parser_nri.add_argument('--decoder', type=str, default='mlp', help='Type of decoder model (mlp, rnn, or sim).')
    parser_nri.add_argument('--decoder-dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser_nri.add_argument('--decoder-hidden', type=int, default=64, help='Number of hidden units.')
    parser_nri.add_argument('--edge-types', type=int, default=2, help='The number of edge types to infer.')
    parser_nri.add_argument('--encoder', type=str, default='mlp', help='Type of path encoder model (mlp or cnn).')
    parser_nri.add_argument('--encoder-dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser_nri.add_argument('--encoder-hidden', type=int, default=64, help='Number of hidden units.')
    parser_nri.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
    parser_nri.add_argument('--gamma', type=float, default=0.5, help='LR decay factor.')
    parser_nri.add_argument('--hard', action='store_true', default=False, help='Uses discrete samples in training forward pass.')
    parser_nri.add_argument('--in-dim', type=int, default=4, help='Number of dimension per variate of the MTS')
    parser_nri.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
    parser_nri.add_argument('--lr-decay', type=int, default=200, help='After how epochs to decay LR by a factor of gamma.')
    parser_nri.add_argument('--num-atoms', type=int, default=10, help='Number of atoms in simulation.')
    parser_nri.add_argument('--no-factor', action='store_true', default=False, help='Disables factor graph model.')
    parser_nri.add_argument('--prediction-steps', type=int, default=1, metavar='N', help='Num steps to predict before re-using teacher forcing.')
    parser_nri.add_argument('--prior', action='store_true', default=False, help='Whether to use sparsity prior.')
    parser_nri.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser_nri.add_argument('--skip-first', action='store_true', default=False, help='Skip first edge type in decoder, i.e. it represents no-edge.')
    parser_nri.add_argument('--stationary', action='store_true', default=True, help='Each time series is stationary.')
    parser_nri.add_argument('--suffix', type=str, default='_springs10', help='Suffix for training data (e.g. "_charged".')
    parser_nri.add_argument('--temp', type=float, default=0.5, help='Temperature for Gumbel softmax.')
    parser_nri.add_argument('--var', type=float, default=5e-5, help='Output variance.')

    args, unknown = parser_nri.parse_known_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.factor = not args.no_factor

    print(args)
    
    return args
    args, unknown = parser_nri.parse_known_args()
    args.factor = not args.no_factor
    
    return args