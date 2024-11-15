import numpy as np
from scipy.linalg import hankel, svd

def cadzow(y, target_rank, iterations):
    N = len(y)
    L = N // 2

    T = np.conj(hankel(y[:N - L], y[N - L - 1:]))
    Tr = T.copy()

    for iter in range(iterations):
        U, S, Vh = svd(Tr, full_matrices=False)
        r = np.sum(S > np.max(S) * max(T.shape) * np.finfo(S.dtype).eps)
        
        if r <= target_rank:
            break
        else:
            S[target_rank:] = 0
            Tr = (U * S) @ Vh
            Tr = avg_hankel(Tr)

    yOut = np.conj(np.concatenate([Tr[:, 0], Tr[-1, 1:]]))
    return yOut

def avg_hankel(matrix):
    avg_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[1]):
        avg_matrix[:, i] = np.mean(matrix.diagonal(i - matrix.shape[0] + 1))
    return avg_matrix
