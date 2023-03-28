# DL_Group9
Paper reproduction
## Results to reproduce
Table 1; Fig. 4; OASIS-1 only
## data
Yes, apply for OASIS-1 and OASIS-2: https://www.oasis-brains.org/#access


# to run plot_nii.py
cd ./NODEO-DIR/
python3 ../plot_nii.py


# to configure lambda1 in table 1
python3 Registration.py --lambda_J 2

# download data and process
https://oasis-brains.org/
go to download
right hand: FreeSurfer data, dowanload one(9GB) and extract: tar -xf <filename>
about 40 subfolders, each having a mri folder in it
in mri/orig/ original mgz files can be found
mgz files can be loaded with nib.load
```python
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

path = "./003.mgz"
X = nib.load(path)
X = X.get_fdata()
print("X.shape=",np.shape(X))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X[:,:,5*i])
    plt.gcf().set_size_inches(10, 10)
plt.show()

```
