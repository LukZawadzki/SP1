import numpy as np

def Gaussian_kernel(l=5, sig =1):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l, dtype = np.float32)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)