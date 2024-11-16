from fri_tem.iaftem import iafTEM
import numpy as np
from numpy.linalg import pinv
from fri_tem.cadzow import cadzow
from scipy.linalg import hankel, svd
from numpy.linalg import pinv

def sepctrum(signal,K, Kmax,b,d,kappa,T):
    y = 0
    m = np.arange(-Kmax, Kmax + 1)
    G = np.zeros(len(m), dtype=complex)
    N = len(signal)
    t = np.linspace(0, 1 - 1 / N, N)
    dt = 1 / N

    for n in range(N):
        G += signal[n] * np.exp(-2 * np.pi * 1j * n * m / N)

    y = np.sum([G[i] * np.exp(1j * m[i] * 2 * np.pi / T * t) for i in range(len(m))], axis=0).real / N

    tnIdx, yInt = iafTEM(y, dt, b, d, kappa)
    tn = t[tnIdx]
    print(tn)
    Ntn = len(tn)
    print(Ntn)
    yDel = -b * np.diff(tn) + kappa * d

    # Further calculations
    w0 = 2 * np.pi / T
    F = np.exp(1j * w0 * np.outer(tn[1:], np.arange(-K, K + 1))) - np.exp(1j * w0 * np.outer(tn[:-1], np.arange(-K, K + 1)))
    F[:, K] = tn[1:] - tn[:-1]
    s = T / (1j * 2 * np.pi * np.arange(-K, K + 1))
    s[K] = 1
    S = np.diag(s)

    ytnHat = pinv(F @ S) @ yDel
    ytnHat = ytnHat.conj().T * N

    spectrum = ytnHat[K:].conj()
    spectrum_denoised = cadzow(spectrum, target_rank=Kmax, iterations=K*100)
    return tn, spectrum_denoised