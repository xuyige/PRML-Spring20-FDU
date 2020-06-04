from handout import gmm, data
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--n', type=int, default=3000)
    parser.add_argument('--mean', type=str, default='1,1,3,3,0,4')
    parser.add_argument('--cov', type=str,
                        default='1,0,0,1,0.5,0,0,0.5,0.8,0,0,0.8')
    parser.add_argument('--iter', type=int, default=60)
    parser.add_argument('--k', type=int, default='3')
    parser.add_argument('--pr', type=str, default='0.3333, 0.3333, 0.3333')

    args = parser.parse_args()
    n, n_iter, d , k= args.n, args.iter, args.d, args.k
    mean_str, cov_str, p_str = args.mean, args.cov, args.pr
    n_mean, n_cov, p= [], [], []
    try:
        mean_str = mean_str.split(',')
        for i in range(k):
            n_mean.append([float(mean_str[i * d + j]) for j in range(d)])
    except:
        print('the input of args mean is wrong ')
    try:
        cov_str = cov_str.split(',')
        for i in range(k):
            cov_temp = []
            for j in range(d):
                cov_temp.append([float(cov_str[i * d * d + j * d + k]) for k in range(d)])
            n_cov.append(cov_temp)
    except:
        print('the input of args cov is wrong ')
    try:
        p_str = p_str.split(',')
        for i in range(k):
            p.append(float(p_str[i]))
    except:
        print('the input of args cov is wrong ')

    x, label = data.generate_data(n_samples = n, n_centers = k,
                             cluster_std = n_cov, center_box = n_mean, p = p)
    n = len(label)
    rand_mu = x[np.random.choice(range(n), k)]
    gmm.plot_raw_date2d(x, label, k, n_mean, n_cov)
    model = gmm.gmm(k, n, d, rand_mu)
    model.init_with_kmeans(30, x)
    model.run_EM(n_iter, x, n_mean, n_cov)