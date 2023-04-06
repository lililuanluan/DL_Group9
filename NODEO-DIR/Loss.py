import torch
import torch.nn.functional as F
import numpy as np
from Utils import *

class NCC(torch.nn.Module):
    """
    NCC with cumulative sum implementation for acceleration. local (over window) normalized cross correlation.
    equation(8) similarity loss is L = 1-NCC
    """

    def __init__(self, win=21, eps=1e-5):
        super(NCC, self).__init__()
        self.eps = eps
        self.win = win
        self.win_raw = win

    def window_sum_cs3D(self, I, win_size):
        half_win = int(win_size / 2)
        pad = [half_win + 1, half_win] * 3 # concat 3 same list
        # print("pad=", pad)

        I_padded = F.pad(I, pad=pad, mode='constant', value=0)  # [x+pad, y+pad, z+pad]
        # print("I_padded shape", I_padded.shape)
        # print("I = ", I)
        # print("I_padded = ", I_padded)

        # Run the cumulative sum across all 3 dimensions
        I_cs_x = torch.cumsum(I_padded, dim=2)
        # print("I_cs_x", I_cs_x[-1,-1,:,:,:])
        I_cs_xy = torch.cumsum(I_cs_x, dim=3)
        # print("I_cs_xy", I_cs_xy[-1,-1,:,:,:])
        I_cs_xyz = torch.cumsum(I_cs_xy, dim=4) # sum of all items with less or equal than one's own idex
        # print("I_cs_xyz", I_cs_xyz[-1,-1,:,:,:])

        x, y, z = I.shape[2:] # ignore the first two, squeezed out
        # print("I_cs_xyz.shape", I_cs_xyz.shape)
        # print("I_cs_xyz[:, :, win_size:, win_size:, win_size:]=",I_cs_xyz[-1, -1, win_size:, win_size:, win_size:]) # remove paddings
        # print("I_cs_xyz[:, :, win_size:, win_size:, :z]=",I_cs_xyz[-1, -1, win_size:, win_size:, :z])

        # Use subtraction to calculate the window sum
        I_win = I_cs_xyz[:, :, win_size:, win_size:, win_size:] \
                - I_cs_xyz[:, :, win_size:, win_size:, :z] \
                - I_cs_xyz[:, :, win_size:, :y, win_size:] \
                - I_cs_xyz[:, :, :x, win_size:, win_size:] \
                + I_cs_xyz[:, :, win_size:, :y, :z] \
                + I_cs_xyz[:, :, :x, win_size:, :z] \
                + I_cs_xyz[:, :, :x, :y, win_size:] \
                - I_cs_xyz[:, :, :x, :y, :z]

        return I_win

    def forward(self, I, J):
        # compute CC squares
        I = I.double()
        J = J.double()
        # print("I.shape=", I.shape)

        I2 = I * I
        J2 = J * J
        IJ = I * J

        # print("I*I.shape=", I.shape)

        # compute local sums via cumsum trick
        I_sum_cs = self.window_sum_cs3D(I, self.win)
        J_sum_cs = self.window_sum_cs3D(J, self.win)
        I2_sum_cs = self.window_sum_cs3D(I2, self.win)
        J2_sum_cs = self.window_sum_cs3D(J2, self.win)
        IJ_sum_cs = self.window_sum_cs3D(IJ, self.win)

        win_size_cs = (self.win * 1.) ** 3

        u_I_cs = I_sum_cs / win_size_cs
        u_J_cs = J_sum_cs / win_size_cs

        cross_cs = IJ_sum_cs - u_J_cs * I_sum_cs - u_I_cs * J_sum_cs + u_I_cs * u_J_cs * win_size_cs
        I_var_cs = I2_sum_cs - 2 * u_I_cs * I_sum_cs + u_I_cs * u_I_cs * win_size_cs
        J_var_cs = J2_sum_cs - 2 * u_J_cs * J_sum_cs + u_J_cs * u_J_cs * win_size_cs

        cc_cs = cross_cs * cross_cs / (I_var_cs * J_var_cs + self.eps)
        cc2 = cc_cs  # cross correlation squared

        # return negative cc.
        return 1. - torch.mean(cc2).float()

class NCC_2D(torch.nn.Module):
    """
    NCC with cumulative sum implementation for acceleration. local (over window) normalized cross correlation.
    equation(8) similarity loss is L = 1-NCC
    """

    def __init__(self, win=21, eps=1e-5):
        super(NCC_2D, self).__init__()
        self.eps = eps
        self.win = win
        self.win_raw = win

    def window_sum_cs3D(self, I, win_size):
        half_win = int(win_size / 2)
        pad = [half_win + 1, half_win] * 2
        

        I_padded = F.pad(I, pad=pad, mode='constant', value=0)  # [x+pad, y+pad, z+pad]

        # Run the cumulative sum across all 3 dimensions
        I_cs_x = torch.cumsum(I_padded, dim=2)
        I_cs_xy = torch.cumsum(I_cs_x, dim=3)

        x, y = I.shape[2:]

        # Use subtraction to calculate the window sum
        I_win = I_cs_xy[:, :, win_size:, win_size:] \
                - I_cs_xy[:, :, win_size:, :y] \
                - I_cs_xy[:, :, :x, win_size:] \
                + I_cs_xy[:, :, :x, :y]

        return I_win

    def forward(self, I, J):
        # compute CC squares
        I = I.double()
        J = J.double()

        I2 = I * I

        J2 = J * J
        IJ = I * J

        # equation 8
        # Sum(I-I_bar)(J-J_bar) = Sum(IJ - I*J_bar - I_bar*J -I_bar*J_bar)
        #                        IJ_sum_cs


        # compute local sums via cumsum trick
        I_sum_cs = self.window_sum_cs3D(I, self.win)
        J_sum_cs = self.window_sum_cs3D(J, self.win)
        I2_sum_cs = self.window_sum_cs3D(I2, self.win)
        J2_sum_cs = self.window_sum_cs3D(J2, self.win)
        IJ_sum_cs = self.window_sum_cs3D(IJ, self.win)

        win_size_cs = (self.win * 1.) ** 2

        u_I_cs = I_sum_cs / win_size_cs
        u_J_cs = J_sum_cs / win_size_cs

        cross_cs = IJ_sum_cs - u_J_cs * I_sum_cs - u_I_cs * J_sum_cs + u_I_cs * u_J_cs * win_size_cs
        I_var_cs = I2_sum_cs - 2 * u_I_cs * I_sum_cs + u_I_cs * u_I_cs * win_size_cs
        J_var_cs = J2_sum_cs - 2 * u_J_cs * J_sum_cs + u_J_cs * u_J_cs * win_size_cs

        cc_cs = cross_cs * cross_cs / (I_var_cs * J_var_cs + self.eps)
        cc2 = cc_cs  # cross correlation squared

        # return negative cc.
        return 1. - torch.mean(cc2).float()

def JacboianDet(J):
    if J.size(-1) != 3:
        J = J.permute(0, 2, 3, 4, 1)
    J = J + 1
    J = J / 2.
    scale_factor = torch.tensor([J.size(1), J.size(2), J.size(3)]).to(J).view(1, 1, 1, 1, 3) * 1.
    J = J * scale_factor

    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
    Jdet1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 0])
    Jdet2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy[:, :, :, :, 1] * dz[:, :, :, :, 0])

    Jdet = Jdet0 - Jdet1 + Jdet2
    # print("Jdet=", Jdet)
    return Jdet

def neg_Jdet_loss(J):
    Jdet = JacboianDet(J)
    neg_Jdet = -1.0 * (Jdet - 0.5)
    selected_neg_Jdet = F.relu(neg_Jdet)
    return torch.mean(selected_neg_Jdet ** 2)
    # return torch.log10(torch.mean(selected_neg_Jdet ** 2))

def smoothloss_loss(df):
    return (((df[:, :, 1:, :, :] - df[:, :, :-1, :, :]) ** 2).mean() + \
     ((df[:, :, :, 1:, :] - df[:, :, :, :-1, :]) ** 2).mean() + \
     ((df[:, :, :, :, 1:] - df[:, :, :, :, :-1]) ** 2).mean())

def magnitude_loss(all_v):
    all_v_x_2 = all_v[:, :, 0, :, :, :] * all_v[:, :, 0, :, :, :]
    all_v_y_2 = all_v[:, :, 1, :, :, :] * all_v[:, :, 1, :, :, :]
    all_v_z_2 = all_v[:, :, 2, :, :, :] * all_v[:, :, 2, :, :, :]
    all_v_magnitude = torch.mean(all_v_x_2 + all_v_y_2 + all_v_z_2)
    return all_v_magnitude

def test_ncc():
    

    # -----------------debugging 3D----------------
    # fixed_path = '../data-sample/images/aligned_norm.nii.gz'
    # fixed = load_nii_2(fixed_path)
    # # print("fixed shape=", fixed)
    # # fixed = np.random.randint(0, high=10, size=(2,2,2))
    # print("fixed=", fixed)
    # # fixed = fixed[:,:,1]
    # # print("fixed(one slice) shape=", fixed.shape)
    # device = torch.device('cuda:0')
    # fixed = torch.from_numpy(fixed).to(device).float()
    # print("fixed(to tensor) shape=", fixed.shape)
    # fixed = fixed.unsqueeze(0).unsqueeze(0)
    # print("fixed(unsqueeze) shape=", fixed.shape)
    # loss_NCC = NCC(win=1) # also mentioned in the paper
    # loss_sim = loss_NCC(fixed, fixed)
    # print("loss_sim", loss_sim)


    # -----------------testing 2D----------------
    fixed_path = '../data-sample/images/aligned_norm.nii.gz'
    fixed = load_nii_2(fixed_path)
    print("fixed shape=", fixed.shape)
    fixed = fixed[:,:,1]
    print("fixed(one slice) shape=", fixed.shape)
    device = torch.device('cuda:0')
    fixed = torch.from_numpy(fixed).to(device).float()
    print("fixed(to tensor) shape=", fixed.shape)
    fixed = fixed.unsqueeze(0).unsqueeze(0)
    print("fixed(unsqueeze) shape=", fixed.shape)
    loss_NCC = NCC_2D(win=21) # also mentioned in the paper
    loss_sim = loss_NCC(fixed, fixed)
    print("loss_sim", loss_sim)

if __name__ == '__main__':
    test_ncc()