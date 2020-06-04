import random
import numpy as np
import matplotlib.pyplot as plt

class Dataset():
    def __init__(self, K=3, mean=[[1,1], [5,5], [9,9]], 
                    cov=[[[1,0], [0,1]], [[1,0], [0,1]], [[1,0], [0,1]]],
                    size=[1000, 1000, 1000],
                    path = 'a.data'):
        mean = np.array(mean)
        cov = np.array(cov)
        if mean.shape[0] != K or cov.shape[0] != K or len(size) != K:
            print('Init failed')
            exit(0)
        self.dataWithLabel = []
        self.data = []
        for i in range(K):
            distribution = np.random.multivariate_normal(mean[i],cov[i],size[i])
            for j in range(size[i]):
                self.data.append((distribution.T[0][j], distribution.T[1][j]))
                self.dataWithLabel.append((distribution.T[0][j], distribution.T[1][j], i))

        random.shuffle(self.data)
        with open(path, 'w') as f:
            for x, y in self.data:
                f.write(f'{x} {y}\n')

        print('data generated successfully')

    def plot(self):
        plt.plot([p[0] for p in self.data], [p[1] for p in self.data], '.')
        plt.show()

if __name__ == "__main__":
    data = Dataset()
    data.plot()

