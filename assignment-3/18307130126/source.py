import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

priors = [0.2, 0.25, 0.15, 0.15,0.25]
mu = [[0.7,0.7],[0.8,-0.7],[-0.8,0.9],[0,2],[-1,-1]]
# mu = [[1,1],[1,-1],[-1,1],[0,3],[-1,-1]]
sigma = [[
    [0.3,0.1],
    [0.1,0.2]
], [
    [0.1,0],
    [0,0.3]
], [
    [0.3,0],
    [0,0.3]
], [
    [0.3,0],
    [0,0.1]
], [
    [0.3,0.1],
    [0.1,0.4]
]]


def draw(s):
    plt.scatter(s[:,0], s[:,1], alpha=0.4)

def gen(args):
    
    print(args.number)
    n = args.number
    n_sample = (np.array(priors)*n).astype(int)
    np.random.seed(233)

    s = []
    l = []
    for i in range(5):
        x = np.random.multivariate_normal(mu[i], sigma[i], n_sample[i])
        s.append(x)
        l.append(np.array([i] * n_sample[i]).reshape(1,-1))

    sample = np.concatenate((s[0],s[1],s[2],s[3],s[4]), axis=0)
    label = np.concatenate((l[0],l[1],l[2],l[3],l[4]), axis=1)

    print(sample)
    print(label)

    if args.plot == '1':
        for i in range(5):
            draw(s[i])
        plt.show()

        draw(sample)
        plt.show()
    with open("data.data", "w") as f:
        np.savetxt(f, sample)
    
    with open("label.data", "w") as f2:
        np.savetxt(f2, label)
    return 

def calc_label(sample, knum, mu):
    for i in range(knum):
        now = ((sample - mu[i]) ** 2).sum(axis=1)
        now = now.reshape(1,-1)
        if i == 0:
            ori = now
        else :
            ori = np.concatenate((ori,now), axis=0)
    label = np.argmin(ori, axis=0)
    return label

def dis_minmax(points):
    L = len(points)
    res = sum((points[0]-points[1]) ** 2)
    for i in range(L):
        for j in range(i):
            res = min(res, sum((points[i]-points[j]) ** 2))
    return res

def k_means(sample, knum, kstep):
    L = len(sample)
    mi = 0
    for i in range(10):
        points = sample[np.random.choice(L, knum)]
        for _ in range(kstep):
            label = calc_label(sample, knum, points)
            for i in range(knum):
                points[i] = np.average(sample[label==i], axis=0)
        if i == 0:
            mi = dis_minmax(points)
            ans = points
        elif mi < dis_minmax(points):
            ans = points
            mi = dis_minmax(points)
    return ans

def Gauss(sample, pi, mu, sigma, knum):
    var = [multivariate_normal(mu[i], sigma[i]) for i in range(knum)]
    for i in range(knum):
        now = pi[i] * var[i].pdf(sample)
        now = now.reshape(1,-1)
        if i == 0:
            ori = now
        else :
            ori = np.concatenate((ori,now), axis=0)
    label = np.argmax(ori, axis=0)
    return ori, label 

def EM(sample, knum, kstep, estep, plot, test):
    L = len(sample)

    mu = k_means(sample, knum, kstep)
    label = calc_label(sample, knum, mu)

    if test == '1':
        from itertools import permutations
        with open("label.data","r") as f2:
            real_label = np.loadtxt(f2)
        mx = 0
        for j in list(permutations([0, 1, 2, 3, 4])):
            acc = 0
            for i in range(len(label)):
                if label[i] == j[int(real_label[i])]:
                    acc += 1
            mx = max(mx, acc)
        mx /= len(label)
        print('k-means acc:',mx)
    
    if plot == '1':
        for i in range(knum):
            draw(sample[label==i])
        plt.scatter(mu[:,0], mu[:,1], marker='+', s=60)
        plt.show()

    pi = []
    for i in range(knum):
        pi.append(sum(label==i))
    pi = np.array(pi)
    pi = pi/len(label)
    # print(pi)
    
    sigma = np.array([np.diag(np.var(sample[label==i], axis=0)) for i in range(knum)])
    # print(sigma)

    for i in range(estep):
        res, label = Gauss(sample, pi, mu, sigma, knum)
        gamma = res / res.sum(axis=0).reshape(1,-1)

        Nk = gamma.sum(axis=1)
        
        PIk = Nk / Nk.sum()
        mu = (gamma @ sample)/ Nk.reshape(-1,1) # k by 2

        for k in range(knum):
            for n in range(L):
                if n == 0:
                    # print(gamma[k][n])
                    # print((sample[n] - mu[k]))
                    res = gamma[k][n] * ((sample[n] - mu[k]).reshape(-1,1) @ (sample[n] - mu[k]).reshape(1,-1))
                    # print(res)
                else :
                    res = res + gamma[k][n] * ((sample[n] - mu[k]).reshape(-1,1) @ (sample[n] - mu[k]).reshape(1,-1))
            # print(res.shape)
            sigma[k] = res / Nk[k] 
        # print(sigma)


    if plot == '1':
        for i in range(knum):
            draw(sample[label==i])
        plt.scatter(mu[:,0], mu[:,1], marker='+', s=60)
        plt.show()

    if test == '1':
        from itertools import permutations
        with open("label.data","r") as f2:
            real_label = np.loadtxt(f2)
        mx = 0
        for j in list(permutations([0, 1, 2, 3, 4])):
            acc = 0
            for i in range(len(label)):
                if label[i] == j[int(real_label[i])]:
                    acc += 1
            mx = max(mx, acc)
        mx /= len(label)
        print('EM acc:',mx)

    return pi, mu, sigma

def run(args):
    file = args.file
    with open(file, "r") as f:
        sample = np.loadtxt(f)
    # print(sample)
    pi, mu, sigma = EM(sample, args.knum, args.kstep, args.estep, args.plot, args.test)

    print("pi:")
    print(pi)
    print("mu:")
    print(mu)
    print("sigma:")
    print(sigma)
    # print(args.plot,args.estep,args.kstep,args.acc,args.knum)
    return 

def rundefault():
    with open("data.data", "r") as f:
        sample = np.loadtxt(f)
    # print(sample)
    pi, mu, sigma = EM(sample, 5, 10, 50, '1', '0')
    print("pi:")
    print(pi)
    print("mu:")
    print(mu)
    print("sigma:")
    print(sigma)
    return 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_a = subparsers.add_parser('gen')
    parser_a.set_defaults(func=gen)
    parser_a.add_argument('-n', '--number', type=int, default=100)
    parser_a.add_argument('-p','--plot', choices=['1','0'], default='0')

    parser_b = subparsers.add_parser('run')
    parser_b.set_defaults(func=run)
    parser_b.add_argument('-e','--estep', type=int, default=50)
    parser_b.add_argument('-i','--kstep', type=int, default=10)
    parser_b.add_argument('-p','--plot', choices=['1','0'], default='0')
    parser_b.add_argument('-a','--acc', choices=['1','0'], default='0')
    parser_b.add_argument('-k','--knum', type=int, default=5)
    parser_b.add_argument('-f','--file', type=str, default='data.data')
    parser_b.add_argument('-t','--test', choices=['1','0'], default='0')
    args = parser.parse_args()
    
    try: 
        args.func(args)
    except AttributeError:
        rundefault()