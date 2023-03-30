import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from os import walk
import glob
import skimage.transform as skTrans

path = "./images/*.gz"
files = glob.glob(path)
for path in files:
    print("filename=",path)
    X = nib.load(path)
    
    X = X.get_fdata()
    # X = skTrans.resize(X, (160, 192, 144), order=1, preserve_range=True)
    print(X)
    print("X.shape=",np.shape(X))

    for i in range(29):
        plt.subplot(5, 6, i + 1)
        plt.imshow(X[:,:,5*i])
        plt.gcf().set_size_inches(10, 10)
    plt.show()