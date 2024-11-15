import numpy as np 

def iafTEM(signal, dt, b, d, k):
    Nx = len(signal)
    y_out = []
    tnIdx = []
    y = 0
    j = 0

    for i in range(Nx):
        y += dt * (b + signal[i]) / k
        y_out.append(y)
        
        if y >= d:
            tnIdx.append(i)
            j += 1
            y -= d

    return np.array(tnIdx), np.array(y_out)
