import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from os import walk
import glob
import skimage.transform as skTrans

# path = "./images/*.gz"
path = "../NODEO-DIR/result/warped.nii.gz"
# path = "/home/liluan/桌面/DL_Group9/data-sample/images/2D/002slice/slice_norm.nii.gz"
files = glob.glob(path)
for path in files:
    print("filename=",path)
    X = nib.load(path)
    
    X = X.get_fdata()
    # X = skTrans.resize(X, (160, 192, 144), order=1, preserve_range=True)
    # print(X)
    # X = X[:,:,0,0]
    print("X.shape=",np.shape(X))
    X=X[0,0,:,:]
    plt.imshow(X)
    # for i in range(29):
    #     plt.subplot(5, 6, i + 1)
    #     plt.imshow(X[:,:,5*i])
    #     plt.gcf().set_size_inches(10, 10)
    plt.show()