import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def load_nii(path):
    print("load_nii path=", path)
    X = nib.load(path)
    X = X.get_fdata()
    print("X.shape=",np.shape(X))
    return X

def plot_nii_3(path):
    test_load = load_nii(path)
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(test_load[:,:,5*i])
        plt.gcf().set_size_inches(10, 10)
    plt.show()


def plot_nii_5(path):
    test_load = load_nii(path)
    for j in range(3):
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(test_load[:,:,5*i,0,j])
            plt.gcf().set_size_inches(10, 10)
        plt.show()


plot_nii_3("../OASIS-data/oasis_cs_freesurfer_disc2.tar.gz")

plot_nii_3("./data/OAS1_0001_MR1/brain.nii.gz")
plot_nii_3("./data/OAS1_0001_MR1/brain_aseg.nii.gz")
plot_nii_3("./data/OAS1_0002_MR1/brain.nii.gz")
plot_nii_5("./result/df.nii.gz")
plot_nii_3("./result/warped.nii.gz")