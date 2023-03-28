import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from os import walk
import glob
path = "*.mgz"
files = glob.glob(path)
for path in files:
    print("filename=",path)
    X = nib.load(path)
    X = X.get_fdata()
    print("X.shape=",np.shape(X))

    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X[:,:,5*i])
        plt.gcf().set_size_inches(10, 10)
    plt.show()