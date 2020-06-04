import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import multivariate_normal, norm

colorNames = ['black', 'brown', 'red', 'orange', 'olive', 'yellow', 'green', 'teal', 'cyan', 'deepskyblue',
              'slategray', 'blue', 'royalblue', 'blueviolet', 'violet', 'purple', 'deeppink']

class gaussianMixtureModel():
    def __init__(self, parser):
        super().__init__()
        self.numOfDistributions = parser.numOfDistributions
        self.dimOfDistributions = parser.dimOfDistributions
        self.means = np.zeros([self.numOfDistributions, self.numOfDistributions])
        self.covs = np.zeros([self.numOfDistributions, self.dimOfDistributions, self.dimOfDistributions])
        self.probabilities = np.zeros(self.numOfDistributions)
        self.initializer = kMeansModel(parser)
        indices = np.arange(len(colorNames))
        np.random.shuffle(indices)
        self.plotColors = [colorNames[index] for index in indices[0: self.numOfDistributions]]

    def __call__(self, dataset):
        probabilities = np.zeros([dataset.shape[0], self.numOfDistributions])
        for i in range(self.numOfDistributions):
            mean, cov, probability = self.means[i], self.covs[i], self.probabilities[i]
            probabilities[:, i] = probability * multivariate_normal.pdf(dataset, mean, cov)
        return probabilities

    def fit(self, parser, dataset):
        self.initializer.fit(parser, dataset)
        self.means, self.covs, self.probabilities = self.initializer.getParameters()
        if parser.plot:
            plt.close('all')
            fig = plt.figure()
            plt.ion()

        if parser.epsilonSpecified:
            epsilon = parser.epsilon
            probabilities = self.__call__(dataset)
            delta = 1
            logP = np.sum(np.log(np.sum(probabilities, axis=1)))
            while delta > epsilon:
                if parser.plot:
                    plt.show()
                    self.plot(dataset)
                self.expectationMaximumStep(dataset)
                probabilities = self.__call__(dataset)
                logP1 = np.sum(np.log(np.sum(probabilities, axis=1)))
                delta = logP1 - logP
                logP = logP1
                print(logP, delta)
        elif parser.epochsSpecified:
            epochs = parser.epochs
            for epoch in range(epochs):
                print(epoch)
                if parser.plot:
                    plt.show()
                    self.plot(dataset)
                self.expectationMaximumStep(dataset)
        else:
            raise ValueError('Training method must be specified in command line options.')
        
        if parser.plot:
            plt.ioff()
            fig = plt.gcf()
            plt.savefig(parser.plotPath + 'GMM.jpeg', dpi=800, quality=95)
            plt.show()

    def evaluate(self, dataset, labels):
        probabilities = self(dataset)
        labels1 = np.argmax(probabilities, axis=1)
        proportions, proportions1 = np.zeros(self.numOfDistributions), np.zeros(self.numOfDistributions)
        for i in range(self.numOfDistributions):
            proportions[i] = np.mean(labels == i)
            proportions1[i] = np.mean(labels1 == i)

        labels2 = np.zeros_like(labels)
        for i in range(self.numOfDistributions):
            index, index1 = np.argmax(proportions), np.argmax(proportions1)
            indices = np.where(labels1 == index1)
            proportions[index], proportions1[index1] = -1, -1
            labels2[indices] = index

        accuracy = np.mean(labels2 == labels)
        print('Accuracy:', accuracy)

    def plot(self, dataset):
        probabilities = self.__call__(dataset)
        labels = np.argmax(probabilities, axis=1)
        indices = []
        for i in range(self.numOfDistributions):
            [index] = np.where(labels == i)
            indices.append(index)

        plt.cla()
        if dataset.shape[1] > 1:
            dataset = dataset[:, 0:2]
            means, covs = self.means[:, 0:2], self.covs[:, 0:2, 0:2]
            ax = plt.gca()
            for i in range(self.numOfDistributions):
                index = indices[i]
                if index is not None:
                    mean, cov = means[i], covs[i]
                    vals, vecs = np.linalg.eigh(cov)
                    order = vals.argsort()[::-1]
                    vals, vecs = vals[order], vecs[:, order]
                    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                    width, height = 2 * np.sqrt(vals)
                    ellipse = patches.Ellipse(xy=mean, width=2 * width, height=2 * height, angle=theta, fill=False,
                                              color=self.plotColors[i])
                    plt.scatter(dataset[index, 0], dataset[index, 1], c=self.plotColors[i], s=1)
                    ax.add_artist(ellipse)
        else:
            means, covs = self.means, self.covs
            for i in range(self.numOfDistributions):
                index = indices[i]
                if index is not None:
                    mean, [cov] = means[i], covs[i]
                    std = np.sqrt(cov)
                    xs = np.linspace(mean - 3 * std, mean + 3 * std, 50)
                    ys = norm.pdf((xs - mean) / std)
                    plt.scatter(dataset[index], np.zeros_like(dataset[index]), c=self.plotColors[i], s=1)
                    plt.plot(xs, ys, c=self.plotColors[i])
        plt.axis('auto')
        plt.title('Gaussian Mixture Model')

    def getParameters(self):  
        return self.means, self.covs, self.probabilities

    def expectationMaximumStep(self, dataset):
        N = dataset.shape[0]
        gamma = np.zeros([N, self.numOfDistributions])
        for i in range(self.numOfDistributions):
            mean, cov, probability = self.means[i], self.covs[i], self.probabilities[i]
            gamma[:, i] = probability * multivariate_normal.pdf(dataset, mean, cov)

        gamma = (gamma.transpose() / np.sum(gamma, axis=1)).transpose()
        Nk = np.sum(gamma, axis=0)
        means = self.means
        self.probabilities = Nk / N
        self.means = np.matmul(gamma.transpose(), dataset)
        self.means = (self.means.transpose() / Nk).transpose()
        for i in range(self.numOfDistributions):
            delta = dataset - means[i]
            modifiedDelta = (np.sqrt(gamma[:, i]) * delta.transpose()).transpose()
            self.covs[i] = np.matmul(modifiedDelta.transpose(), modifiedDelta) / Nk[i]

            
class kMeansModel:
    def __init__(self, parser):
        super().__init__()
        self.numOfDistributions = parser.numOfDistributions
        self.dimOfDistributions = parser.dimOfDistributions
        self.means = np.zeros([self.numOfDistributions, self.dimOfDistributions])
        self.covs = np.zeros([self.numOfDistributions, self.dimOfDistributions, self.dimOfDistributions])
        self.probabilities = np.zeros(self.numOfDistributions)
        indices = np.arange(len(colorNames))
        np.random.shuffle(indices)
        self.plotColors = [colorNames[index] for index in indices[0: self.numOfDistributions]]

    def __call__(self, dataset):
        distance = np.zeros([dataset.shape[0], self.numOfDistributions])
        for i in range(self.numOfDistributions):
            center = self.means[i]
            distance[:, i] = np.sqrt(np.sum(np.power(dataset - center, 2), axis=1))
        labels = np.argmin(distance, axis=1)
        return labels

    def fit(self, parser, dataset):
        assert(dataset.shape[0] > self.numOfDistributions)
        num, dim = self.numOfDistributions, self.dimOfDistributions
        self.means[0] = dataset[np.random.randint(dataset.shape[0])]
        for i in range(num - 1):
            distances = np.sum(np.vstack([np.sqrt(np.sum(np.power(dataset - center, 2), axis=1)).transpose()
                                          for center in self.means]), axis=0)
            index = np.argmax(distances)
            self.means[i + 1] = dataset[index]

        if parser.plot:
            plt.close('all')
            fig = plt.figure()
            plt.ion()

        labels = self(dataset)
        labels1 = np.zeros_like(labels)
        while not np.all(labels == labels1):
            labels1 = labels
            for i in range(num):
                indices = np.where(labels1 == i)
                if indices is not None:
                    self.means[i] = np.mean(dataset[indices], axis=0)
                    if self.dimOfDistributions == 1:
                        self.covs[i] = np.cov(np.squeeze(dataset[indices]))
                    else:
                        self.covs[i] = np.cov(dataset[indices].transpose())
            labels = self(dataset)
            if parser.plot:
                plt.show()
                self.plot(dataset)

        n = dataset.shape[0]
        for i in range(num):
            indices = np.where(labels == i)
            self.probabilities[i] = len(indices[0]) / n
            self.covs[i] = np.cov(dataset[indices].transpose())

        if parser.plot:
            plt.ioff()
            fig=  plt.gcf()
            fig.savefig(parser.plotPath + 'KMM.jpeg', dpi=800, quality=95)
            plt.show()

    def evaluate(self, dataset, labels):
        labels1 = self(dataset)
        proportions, proportions1 = np.zeros(self.numOfDistributions), np.zeros(self.numOfDistributions)
        for i in range(self.numOfDistributions):
            proportions[i] = np.mean(labels == i)
            proportions1[i] = np.mean(labels1 == i)

        labels2 = np.zeros_like(labels)
        for i in range(self.numOfDistributions):
            index, index1 = np.argmax(proportions), np.argmax(proportions1)
            indices = np.where(labels1 == index1)
            proportions[index], proportions1[index1] = -1, -1
            labels2[indices] = index

        accuracy = np.mean(labels2 == labels)
        print('Accuracy:', accuracy)

    def plot(self, dataset):
        labels = self.__call__(dataset)
        indices = []
        for i in range(self.numOfDistributions):
            [index] = np.where(labels == i)
            indices.append(index)

        plt.cla()
        if dataset.shape[1] > 1:
            dataset = dataset[:, 0:2]
            means, covs = self.means[:, 0:2], self.covs[:, 0:2, 0:2]
            ax = plt.gca()
            for i in range(self.numOfDistributions):
                index = indices[i]
                if index is not None:
                    mean, cov = means[i], covs[i]
                    vals, vecs = np.linalg.eigh(cov)
                    order = vals.argsort()[::-1]
                    vals, vecs = vals[order], vecs[:, order]
                    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                    width, height = 2 * np.sqrt(vals)
                    ellipse = patches.Ellipse(xy=mean, width=2 * width, height=2 * height, angle=theta, fill=False,
                                              color=self.plotColors[i])
                    plt.scatter(dataset[index, 0], dataset[index, 1], c=self.plotColors[i], s=1)
                    ax.add_artist(ellipse)
        else:
            means, covs = self.means, self.covs
            for i in range(self.numOfDistributions):
                index = indices[i]
                if index is not None:
                    mean, [cov] = means[i], covs[i]
                    std = np.sqrt(cov)
                    xs = np.linspace(mean - 3 * std, mean + 3 * std, 50)
                    ys = norm.pdf((xs - mean) / std)
                    plt.scatter(dataset[index], np.zeros_like(dataset[index]), c=self.plotColors[i], s=1)
                    plt.plot(xs, ys, c=self.plotColors[i])
        plt.axis('auto')
        plt.title('K-Means Model')

    def getParameters(self):  
        return self.means, self.covs, self.probabilities


def trainModel(parser):
    dataset = pickle.load(open(parser.datasetPath, 'rb'))
    if parser.resume:
        if parser.model == 'GMM':
            model = pickle.load(open(parser.resumeModelPath + 'trainedGaussianMixtureModel.pkl', 'rb'))
        elif parser.model == 'KMM':
            model = pickle.load(open(parser.resumeModelPath + 'trainedKMeansModel.pkl', 'rb'))
        else:
            raise RuntimeError
    else:
        if parser.model == 'GMM':
            model = gaussianMixtureModel(parser)
        elif parser.model == 'KMM':
            model = kMeansModel(parser)
        else:
            raise RuntimeError

    model.fit(parser, dataset)
    if parser.saveModel:
        if parser.model == 'GMM':
            pickle.dump(model, open(parser.saveModelPath + 'trainedGaussianMixtureModel.pkl', 'wb'))
        elif parser.model == 'KMM':
            pickle.dump(model, open(parser.saveModelPath + 'trainedKMeansModel.pkl', 'wb'))
        else:
            raise RuntimeError

    return model


def evaluateModel(parser):
    dataset = pickle.load(open(parser.datasetPath, 'rb'))
    labels = pickle.load(open(parser.labelsPath, 'rb'))

    if parser.model == 'GMM':
        model = pickle.load(open(parser.saveModelPath + 'trainedGaussianMixtureModel.pkl', 'rb'))
    elif parser.model == 'KMM':
        model = pickle.load(open(parser.saveModelPath + 'trainedKMeansModel.pkl', 'rb'))
    else:
        raise RuntimeError

    model.evaluate(dataset, labels)
    return model
