import scipy.special
import numpy as np

def shapley_kernel(M, s):
    if s == 0 or s == M:
        return 10000
    t1 = scipy.special.binom(M, s)  # 从M个里面挑s个有多少个组合,C(m,s)
    t2 = t1 * s * (M - s)
    return (M-1) / t2
    return (M - 1) / (scipy.special.binom(M, s) * s * (M - s))


if __name__ == '__main__':
    train_data = np.random.rand(3, 28, 28)
    mean = np.mean(train_data, axis=0)
    mean1 = np.mean(train_data, axis=1)

    train_data -= mean


    # for i in range(16):
    #     print(shapley_kernel(15, i))