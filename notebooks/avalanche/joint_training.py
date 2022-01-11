"""
This is a simple example to show how a simple "offline" upper bound can be
computed. This is useful to see what's the maximum accuracy a model can get
without the hindering of learning continually. This is often referred to as
"cumulative", "joint-training" or "offline" upper bound.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.models import SimpleMLP
from avalanche.training.strategies import JointTraining

import os

if 'TOY_DATA_DIR' not in os.environ: 
    os.environ['TOY_DATA_DIR'] = "/media/data_cifs/projects/prj_fossils/data/toy_data"
        
default_root_dir = os.environ['TOY_DATA_DIR']


def main(args):

    # Config
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")
    # model
    model = SimpleMLP(num_classes=10)

    # CL Benchmark Creation
    perm_mnist = PermutedMNIST(dataset_root=default_root_dir, n_experiences=5)
    train_stream = perm_mnist.train_stream
    test_stream = perm_mnist.test_stream

    # Prepare for training & testing
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    # Joint training strategy
    joint_train = JointTraining(
        model, optimizer, criterion, train_mb_size=32, train_epochs=1,
        eval_mb_size=32, device=device)

    # train and test loop
    results = []
    print("Starting training.")
    # Differently from other avalanche strategies, you NEED to call train
    # on the entire stream.
    joint_train.train(train_stream)
    results.append(joint_train.eval(test_stream))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0,
                        help='Select zero-indexed cuda device. -1 to use CPU.')
    args = parser.parse_args()

    main(args)