import numpy as np
def generate_data(n_samples=100, n_centers=2, cluster_std=[[[1,0],[0,1]],[[1,0],[0,1]]],
                  center_box=[[-10.0, 10.0], [0, 0]], p=None):
    data = []
    label = []
    if p == None:
        n_samples_per_center = [int(n_samples // n_centers)] * n_centers
    else:
        n_samples_per_center = (np.asarray(p) * n_samples).astype(int)
    for i, n in enumerate(n_samples_per_center):
        data_i = np.random.multivariate_normal(mean=center_box[i], cov=cluster_std[i], size=n)
        label_i = i * np.ones(n, dtype=int)
        if data == []:
            data = data_i
            label = label_i
        else:
            data = np.concatenate((data, data_i))
            label= np.concatenate((label, label_i))

    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(label)

    return data, label