# Introduction
In this project we reproduce the "[NODEO: A Neural Ordinary Differential Equation Based Optimization Framework for Deformable Image Registration](https://arxiv.org/abs/2108.03443)" paper. The paper proposes a new image registration method based on neural ODE. The original code base provided by the authors only shows the procedure of 3D image registration. Our reproduction consists of two stages: first we apply the provided code for 3D image registrations using OASIS data set, which corresponds to Table 1 in the paper; second we adjust the network structures and loss functions to conduct 2D image registration. 

In our code base, the folder "data-sample" contains new data set that are used for reproducing both 3D and 2D registration. The folder "NODEO-DIR", which is the author's repository name, contains our modified code. The terminal commands used for different experiments are documented in the README file. 

# Reproduction

## Environment setup
We conduct our experiments on Ubuntu 20.04. The latest version of [Pytorch](https://pytorch.org/get-started/locally/) is installed. In order to train the network on GPU, a compatible [Nvidia driver](https://www.nvidia.com/download/index.aspx) and [CUDA](https://linuxhint.com/install-cuda-ubuntu-2004/) must be installed. Although Pytorch supports CPU-only training, it is recomanded to have GPU enabled. All required python packages in the code can be installed using package managers such as [pip](https://pypi.org/project/pip/) or [conda](https://docs.conda.io/en/latest/).  

## New dataset
A New neuroimaging dataset of brain [medical-datasets](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md) from Dalca et. al is used in this reproduction. Based on the [OASIS](https://oasis-brains.org/) dataset, this new dataset is converted to *.nii.gz* format which can be used directly in this reproduction.

In this dataset, each subject directory contains bias-corrected aligned images (`aligned_norm.nii.gz`) which aligned the corrected image from scanner space to template space. A 35-label segmentation (`aligned_seg35`) of major anatomical regions is also provided in each subject.

## New dataset implementation
Subject 1 (OASIS_OAS1_0001_MR1) and subject 2 (OASIS_OAS1_0002_MR1) are chosen in our reproduction.
### Method
We download the dataset from [medical-datasets](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md).
Then the images in dataset need to be cropped to meet the requirement of given code. The size of `aligned_norm.nii.gz` is 160 x 192 x 224, and we need to crop the image to 160 x 192 x 144 via the following comand.
```
# Utils.py/load_nii_2
X = nib.load(path)
X = X.get_fdata()
X = X[:, :, 40:184]
return X
```


Next, match the images in the dataset with the images in code. Aligned corrected image from subject 1 (`aligned_norm.nii.gz`) is treated as fixed image, aligned corrected image from subject 2 (`aligned_norm_2.nii.gz`) is treated as moving image. Correspondingly, the 35-label segmentations of subject 1 and 2 are set to fixed and moving segmentations.

The command below is used to apply the new dataset.
```
python3 Registration.py --fixed ../data-sample/images/aligned_norm.nii.gz --moving ../data-sample/images/aligned_norm_2.nii.gz --fixed_seg ../data-sample/images/aligned_seg35.nii.gz --moving_seg ../data-sample/images/aligned_seg35_2.nii.gz
```

### Results
The experiment results are shown below, with lambda_J configured as 2 and 2.5 respectively. 
![](https://i.imgur.com/uT0gTrA.png)
(lambda_J=2)
![](https://i.imgur.com/c9skTCm.png)
(lambda_J=2.5)

## Code implementaion（3D to 2D)
In this section we modify the code from 3D to 2D implementations. This includes modifying network structures and loss functions. To run 2d registration, after making sure that the input is a 2D image, then add `--twod True` in the command line.



### Network structure
In the original paper, the author used a convolutional neural network  to model the vector field, however, the code that the author provided only support 3D images, the structure of the CNN is as followed.  
![](https://i.imgur.com/Ydxmp9g.png)
To change into 2D images, we simply need to chage the Conv3d function in `Network.py` into `Conv2d` and modify the corresponding parameters.
```
class BrainNet(ODEF):
    def __init__(self, img_sz, smoothing_kernel, smoothing_win, smoothing_pass, ds, bs):
        super(BrainNet, self).__init__()
        padding_mode = 'replicate'
        bias = True
        self.ds = ds
        self.bs = bs
        self.img_sz = img_sz
        self.smoothing_kernel = smoothing_kernel
        self.smoothing_pass = smoothing_pass
        # self.enc_conv1 = nn.Conv3d(3, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv2 = nn.Conv3d(3, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv3 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv4 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv5 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv6 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.bottleneck_sz = int(
            math.ceil(img_sz[0] / pow(2, self.ds)) * math.ceil(img_sz[1] / pow(2, self.ds)) * math.ceil(
                img_sz[2] / pow(2, self.ds)))
        self.lin1 = nn.Linear(864, self.bs, bias=bias)
        self.lin2 = nn.Linear(self.bs, self.bottleneck_sz * 3, bias=bias)
        self.relu = nn.ReLU()

        # Create smoothing kernels
        if self.smoothing_kernel == 'AK':
            self.sk = AveragingKernel(win=smoothing_win)
        else:
            self.sk = GaussianKernel(win=smoothing_win, nsig=0.1)
```
The modified network structure is as followed:
```
class BrainNet_2D(ODEF):
    #input:imgsz:160*192 (3channels)
    def __init__(self, img_sz, smoothing_kernel, smoothing_win, smoothing_pass, ds, bs):
        super(BrainNet_2D, self).__init__()
        padding_mode = 'zeros'
        bias = True
        self.ds = ds
        self.bs = bs
        self.img_sz = img_sz
        self.smoothing_kernel = smoothing_kernel
        self.smoothing_pass = smoothing_pass
        # self.enc_conv1 = nn.Conv3d(3, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv2 = nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.bottleneck_sz = int(
            math.ceil(img_sz[0] / pow(2, self.ds)) * math.ceil(img_sz[1] / pow(2, self.ds)))
        self.lin1 = nn.Linear(288, self.bs, bias=bias)
        self.lin2 = nn.Linear(self.bs, self.bottleneck_sz * 2, bias=bias)
        self.relu = nn.ReLU()

        # Create smoothing kernels
        if self.smoothing_kernel == 'AK':
            self.sk = AveragingKernel_2D(win=smoothing_win)
        else:
            self.sk = GaussianKernel_2D(win=smoothing_win, nsig=0.1)
```
One thing worth mentioning is that after changing to 2D images, the neurons of the first fully connected layer in the dense layers is changed from 864 to 288 to adapt the dimension changes.
Also to adapt the change in dimensions, the averageing kernel and Gaussian kernel used are changed to their 2D-counterparts. We will further discuss them in section 2.3.3.

### Loss function

In the paper, the total loss consists of four terms: `loss_sim`(similarity loss, euqation 8), `loss_v`(panalizing the magnitude of the velocity field, equation 11), `loss_J`(Jacobian determinants loss, equation 10) and `loss_df`(spatial gradients of the transformed voxel cloud, equation 12), where the last three are regularization terms of different adjustable weights.
```
 loss = loss_sim + loss_v + loss_J + loss_df
```
To change the input from 3D to 2D, we need to adjust the dimension of input. Below shows the changes made to four functions in *Loss.py*.

For similarity loss, only the `window_sum_cs2D` function needs to be changed. This function calculates the sumation of surrounding voxels inside  a window for each voxel in the image. The cumsum function is used to avoid nested for loops. Instead, differencing cumulative sums and complementing the overlapping areas will produce the window sums. The 2D version of this fuction is listed below.
```
def window_sum_cs2D(self, I, win_size):
    half_win = int(win_size / 2)
    pad = [half_win + 1, half_win] * 2
    I_padded = F.pad(I, pad=pad, mode='constant', value=0)  # [x+pad, y+pad, z+pad]

    # Run the cumulative sum across all 3 dimensions
    I_cs_x = torch.cumsum(I_padded, dim=2)
    I_cs_xy = torch.cumsum(I_cs_x, dim=3)

    x, y = I.shape[2:]
    #print("x=",x, "y=", y)

    # Use subtraction to calculate the window sum
    I_win = I_cs_xy[:, :, win_size:, win_size:] \
            - I_cs_xy[:, :, win_size:, :y] \
            - I_cs_xy[:, :, :x, win_size:] \
            + I_cs_xy[:, :, :x, :y]

    return I_win
```


For the 2D version of Jacobian determinant loss, first rescale input `J`, calculate the partial derivatives along x and y dimension and then calcuate the determinant of 2×2 Jacobian matrix to get the Jacobian determinant. The modified `JacobianDet(J)` function is shown below.

Based on the Jacobian determinant calculated above, `neg_Jdet_loss(J)` uses ReLU activation function to select negative Jacobian determinant and then calculate the final Jacobian determinant loss. According to the provided code, no need to change this function for 2D version. 

```
def JacobianDet_2D(J):
    if J.size(-1) != 2: #last dimension of J 
        #original 1 x 2 y 
        #print("J.O=", J)
        J = J.permute(0, 2, 3, 1) #transpose dimension 1 y x 2
        #need 1 y x 2
        #print("J.PER=", J)
    J = J + 1
    J = J / 2.
    scale_factor = torch.tensor([J.size(1), J.size(2)]).to(J).view(1, 1, 1, 2) * 1.
    J = J * scale_factor

    dy = J[:, 1:, :-1, :] - J[:, :-1, :-1, :]
    dx = J[:, :-1, 1:, :] - J[:, :-1, :-1, :]

    Jdet0 = dx[:, :, :, 0] * dy[:, :, :, 1] 
    Jdet1 = dx[:, :, :, 1] * dy[:, :, :, 0] 

    Jdet = Jdet0 - Jdet1 
    #print("Jdet=", Jdet)
    return Jdet
```


For the 2D version of smoothness loss, calculate the mean of square difference between neighbour elements along x and y dimensions. We need to remove the term related to the z dimension from 3D version, code is shown below.
```
def smoothloss_loss_2D(df): # df 1 2 x y
    return (((df[:, :, 1:, :] - df[:, :, :-1, :]) ** 2).mean() + \
     ((df[:, :, :, 1:] - df[:, :, :, :-1]) ** 2).mean())
```

For the 2D version of magnitude loss, calculate the mean of squared magnitudes of the displacement vectors in the x and y dimensions. Similar to the previous modification, remove the z dimension part as shown below.
```
def magnitude_loss_2D(all_v):# [1, 1, 2, x, y]
    #print("all_v=", all_v)
    all_v_x_2 = all_v[:, :, 0, :, :] * all_v[:, :, 0, :, :]
    all_v_y_2 = all_v[:, :, 1, :, :] * all_v[:, :, 1, :, :]
    #print("all_v_x_2=", all_v_x_2)
  
    all_v_magnitude = torch.mean(all_v_x_2 + all_v_y_2 )
    return all_v_magnitude
```

### Averaging and Gaussian kernel
After acquiring the vector field using CNN, the author proposed to smooth out the vector field by using either Averaging or Gaussian kernel, however, the author did not provide a 2D version kernel to do so. So we need to write our own version of the 2D smoothing kernel.
For averaging kernel, we simply need to remove the addition and substraction of the third dimension. Our implementation is as followed:
```
class AveragingKernel_2D(torch.nn.Module):
    def __init__(self, win=11):
        super(AveragingKernel_2D, self).__init__()
        self.win = win

    def window_averaging(self, v):
        win_size = self.win
        v = v.double()

        half_win = int(win_size / 2)
        pad = [half_win + 1, half_win] * 2

        v_padded = F.pad(v, pad=pad, mode='constant', value=0)  # [x+pad, y+pad, z+pad]
        # Run the cumulative sum across all 3 dimensions
        v_cs_x = torch.cumsum(v_padded, dim=2)
        v_cs_xy = torch.cumsum(v_cs_x, dim=3)
        #v_cs_xyz = torch.cumsum(v_cs_xy, dim=4)

        x, y = v.shape[2:]

        # Use subtraction to calculate the window sum
        v_win = v_cs_xy[:, :, win_size:, win_size:] \
                - v_cs_xy[:, :, win_size:, :y] \
                - v_cs_xy[:, :, :x, win_size:] \
                + v_cs_xy[:, :, :x, :y]

        # Normalize by number of elements

        v_win = v_win / (win_size ** 2)
        v_win = v_win.float()
        return v_win

    def forward(self, v):
        return self.window_averaging(v)
```
The same goes for Gaussian kernel, we simply needs to remove the last dimension and change the convolution used in the forwad process from Conv3d to Conv2d, the code is as followed:
```
class GaussianKernel_2D(torch.nn.Module):
    def __init__(self, win=11, nsig=0.1):
        super(GaussianKernel_2D, self).__init__()
        self.win = win
        self.nsig = nsig
        kernel_x, kernel_y = self.gkern1D_xy(self.win, self.nsig)
        kernel = kernel_x * kernel_y
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)
        self.register_buffer("kernel", kernel)

    def gkern1D(self, kernlen=None, nsig=None):
        '''
        :param nsig: large nsig gives more freedom(pixels as agents), small nsig is more fluid.
        :return: Returns a 1D Gaussian kernel.
        '''
        x = np.linspace(-nsig, nsig, kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern1d = kern1d / kern1d.sum()
        return torch.tensor(kern1d, requires_grad=False).float()

    def gkern1D_xy(self, kernlen=None, nsig=None):
        """Returns 3 1D Gaussian kernel on xyz direction."""
        kernel_1d = self.gkern1D(kernlen, nsig)
        kernel_x = kernel_1d.view(1, 1, -1, 1)
        kernel_y = kernel_1d.view(1, 1, 1, -1)
        return kernel_x, kernel_y

    def forward(self, x):
        pad = int((self.win - 1) / 2)
        # Apply Gaussian by 3D kernel
        # print(self.kernel.shape)
        x = F.conv2d(x, self.kernel, padding=pad)
        return x
```
## Results
Figure below shows the demonstration of the effect of Gaussian smoothing and the Jacobian determinant loss, which is the same as the figure 4 in the paper.

Column (a) shows registration with Guassian smoothing and Jacobian determinant loss, Column (b) with Guassian smoothing only, Column (c) with averaging smoothing and Jacobian determinant loss and Column (d) with neither of them. 
![](https://i.imgur.com/cFMnzx5.png)



## Discussion
As seen from above figures, the result differences from different configurations are insignificant. Unlike Figure 4 in the paper, where using guassian kernel will make the deformation field strongly twisted, different combination of enabled terms(Jacobian loss and Gaussian smooth) result in similar outputs. The reason of this may relate to the used data sample.  




# Task division
Xu Yang, Lu Junyu and Li Luan collaboratedly contributed to reproduction of 3D registration. Xu Yang wrote 2D version of Jacobian loss, smoothness loss and magnitude loss functions. Lu Junyu wrote the 2D CNN, averaging kernel and Gaussian kernel. Li Luan wrote the 2D similarity loss function. 




