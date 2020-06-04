import sys
import argparse
import numpy as np
from handout.data import Dataset
from handout.model import GMM

def data_generate(args):
    mean = [int(x) for x in args.mean.strip().split(' ')]
    mean = np.reshape(np.array(mean), (args.K,args.dim)).tolist()
    cov = [int(x) for x in args.cov.strip().split(' ')]
    cov = np.reshape(np.array(cov), (args.K, args.dim, args.dim)).tolist()
    size = [int(x) for x in args.size.strip().split(' ')]
    dataset = Dataset(K=args.K, mean=mean, cov=cov, size=size, path=args.path)

def gmm(args):
    data = []
    with open(args.data_path, 'r') as f:
        for line in f:
            data.append([float(x) for x in line.strip().split(' ')])
    # print(data)
    model = GMM(K=args.K, dim=args.dim, data=data, init=args.init)
    model.train(epochs=args.epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PRML assignment3.')
    subparser = parser.add_subparsers(help='some subfunctions')

    gd = subparser.add_parser('generate', help='generate data')
    gd.add_argument('--K', type=int, default=3)
    gd.add_argument('--dim', type=int, default=2)
    gd.add_argument('--size', type=str, default='1000 1000 1000')
    gd.add_argument('--mean', type=str, default='1 1 5 5 9 9')
    gd.add_argument('--cov', type=str, default='1 0 0 1 1 0 0 1 1 0 0 1')
    gd.add_argument('--path', type=str, default='a.data')
    gd.set_defaults(func=data_generate)

    gm = subparser.add_parser('GMM', help='GMM model')
    gm.add_argument('--data_path', type=str, default='a.data')
    gm.add_argument('--K', type=int, default=3)
    gm.add_argument('--dim', type=int, default=2)
    gm.add_argument('--init', type=str, default='kmeans')
    gm.add_argument('--epochs', type=int, default=50)
    gm.set_defaults(func=gmm)

    args = parser.parse_args(sys.argv[1:])
    args.func(args)
