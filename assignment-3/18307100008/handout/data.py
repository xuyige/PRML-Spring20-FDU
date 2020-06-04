import numpy as np


def gen_data(means, covs, scale):
    """

    :param means:
    :param covs:
    :param scale:
    :return:
    """
    data = np.ndarray((0, means.shape[1]))
    for i in range(len(means)):
        data = np.row_stack((data, np.random.multivariate_normal(means[i], covs[i], scale[i])))

    np.random.shuffle(data)
    string = ' '.join(['{:.4f}'] * means.shape[1])
    data = [string.format(*d) for d in data]

    with open('data.data', 'w') as dat:
        dat.write('\n'.join(data))


def load_data(fname='data.data'):
    with open(fname, 'r') as dat:
        data = np.array([np.array(row.split()) for row in dat], dtype=np.float32)
    return data
