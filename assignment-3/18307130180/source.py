"""
Python 3.7.5
"""
import argparse

from handout import *

parser = argparse.ArgumentParser()

dataGroup = parser.add_argument_group('data')
dataGroup.add_argument('--createDataset', type=bool, default=True)
dataGroup.add_argument('--datasetPath', type=str, default='handout/samples.data')
dataGroup.add_argument('--numOfSamples', type=int, default=1000)
dataGroup.add_argument('--numOfDistributions', type=int, default=3)
dataGroup.add_argument('--dimOfDistributions', type=int, default=2)
dataGroup.add_argument('--separationDegree', type=int, default=10)
dataGroup.add_argument('--saveLabels', type=bool, default=True)
dataGroup.add_argument('--labelsPath', type=str, default='handout/labels.pkl')

trainGroup = parser.add_argument_group('train')
trainGroup.add_argument('--model', type=str, default='GMM')
trainGroup.add_argument('--train', type=bool, default=True)
trainGroup.add_argument('--plot', type=bool, default=True)
trainGroup.add_argument('--plotPath', type=str, default='handout/')
trainGroup.add_argument('--epsilonSpecified', type=bool, default=True)
trainGroup.add_argument('--epsilon', type=float, default=1e-3)
trainGroup.add_argument('--epochsSpecified', type=bool, default=False)
trainGroup.add_argument('--epochs', type=int, default=50)
trainGroup.add_argument('--saveModel', type=bool, default=True)
trainGroup.add_argument('--saveModelPath', type=str, default='handout/')
trainGroup.add_argument('--resume', type=bool, default=False)
trainGroup.add_argument('--resumeModelPath', type=str, default='handout/')

evaluateGroup = parser.add_argument_group('evaluate')
evaluateGroup.add_argument('--evaluate', type=bool, default=True)

parser = parser.parse_args()

if __name__ == '__main__':
    if parser.createDataset:
        createDataset(parser)
    if parser.train:
        trainModel(parser)
    if parser.evaluate:
        evaluateModel(parser)
