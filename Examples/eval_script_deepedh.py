import matplotlib.pyplot as plt
import pandas as pd
from SPCSim.data_loaders.rgbd_loaders import NYULoader1
from SPCSim.utils.plot_utils import plot_rgbd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from SPCSim.data_loaders.transient_loaders import TransientGenerator
from SPCSim.postproc.edh_postproc import PostProcEDH
from SPCSim.sensors.dtof import PEDHOptimized
from SPCSim.postproc.metric import rmse, ame, p_inlier
import sys
sys.path.append(".")
import torch
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import numpy as np


# Set correct path to NYUv2 dataset folder
csv_file_name = "/home/kaustubh/datasets/nyu_data/data/nyu2_train.csv"
root = "/home/kaustubh/datasets/nyu_data/"
df = pd.read_csv(csv_file_name, header=None)

Nr, Nc = [64,64]
N_tbins = 1024
tmax = 100
FWHM = 0.317925
device = "cuda"
N_pulses = 50
N_edhbins = 32
seed_val = 43
alpha_sig = 1
alpha_bkg = 10

transform = ToTensor()

# Initialize the NYUv2 data loader, set the index and choose the 
# correct paths to rgb and distance images
nyu_data = NYULoader1(Nr,Nc, folder="train")
idx = 1
rgb_pth = root+df[0][idx]
dist_pth = root+df[1][idx]

# Using the get_data method to get rgbd data for specific rgbd frames path
data = nyu_data.get_data(rgb_pth, dist_pth, rgb_pth, rgb_pth)

# Creating transient generator with laser time period of tmax ns, FWHM and with
# laser time period divided into N_tbins equal time-bins
tr_gen = TransientGenerator(Nr = Nr, Nc = Nc, N_tbins = N_tbins, tmax = tmax, FWHM = FWHM)



# Using the get function to generate the transient
# for a given distance, albedo, intensity, and illumination condition
phi_bar = tr_gen.get_transient(data["gt_dist"], # NOTE: the true distance is in meters and depends on tmax
                               data["albedo"]*1.0/255.0,
                               data["albedo"]*1.0/255.0,
                               torch.tensor(alpha_sig),
                               torch.tensor(alpha_bkg))

# Set row and column of the pixel for which you want to track the CV trajectories
ROW, COL = [0,0]

# Initializing the HEDHBaseClass with desired sensor parameters
spc1 = PEDHOptimized(Nr,
              Nc,
              N_pulses,
              device,
              N_tbins,
              N_edhbins,
              pix_r = ROW,
              pix_c = COL,
              seed = seed_val,
              step_params={
                  "k":3, # Step size gain
                  "decay":0, # Setting decay = 0 allows to update final decay value as a function of N_pulses
                  "mtm":0.8,
                  "min_clip":0.02,
                  "switch_fraction":0.8,
                  "delta_mem": 0.95
              })

# Capture method runs the HEDH binners for N_pulses based on the input true transient (phi_bar)
captured_data1 = spc1.capture(phi_bar)

# Creating class to compute distance estimates and photon density estimates from EDH
postproc = PostProcEDH(Nr, Nc, N_tbins, tmax, device)


pedh_data = captured_data1["edh"]
# print(pedh_data.shape, pedh_data)

oedh_data = captured_data1["oedh"]
# print(oedh_data.shape, oedh_data)


pedh_rho0, _, _, pedh_pred_depth0 = postproc.edh2depth_t(pedh_data[:,:,1:-1], mode=0)
pedh_rho1, _, _, pedh_pred_depth1 = postproc.edh2depth_t(pedh_data[:,:,1:-1], mode=1)
pedh_bin_w_inv_, pedh_bin_idx_, _, pedh_pred_depth = postproc.edh2depth_t(pedh_data[:,:,1:-1], mode=2)


oedh_rho0, _, _, oedh_pred_depth0 = postproc.edh2depth_t(oedh_data[:,:,1:-1], mode=0)
oedh_rho1, _, _, oedh_pred_depth1 = postproc.edh2depth_t(oedh_data[:,:,1:-1], mode=1)
oedh_bin_w_inv_, oedh_bin_idx_, _, oedh_pred_depth = postproc.edh2depth_t(oedh_data[:,:,1:-1], mode=2)

bin_w_inv = pedh_rho1.cpu().numpy()
bin_w_inv = bin_w_inv/(bin_w_inv.mean()+0.000000001)
x = transform(bin_w_inv.copy())
x = Variable(x).unsqueeze(0).unsqueeze(0).to(device).float()

model = torch.jit.load('DeePEDH_SingleBounce_Final.zip')
model = model.to(device)
model = model.eval()

pred_depth_idx, _ = model(x)
deepedh_pred_depth = pred_depth_idx*15.0
deepedh_pred_depth = deepedh_pred_depth.view(Nr, Nc).detach()


#####################################
######### Plotting Results ##########

fig, ax = plt.subplots(1,5, figsize=(20,6))
font_s = 15
font_t = 18
alpha1 = 1
alpha2 = 10

rgb_img = data["rgb"].cpu().numpy() 
gt_dist = data["gt_dist"].cpu().numpy()

print(rgb_img.shape)
print(gt_dist.shape)

ax[0].set_title("Scene Image", fontsize=font_t+2)
ax[0].imshow(rgb_img)
ax[0].axis('off')

temp_txt = "RMSE    |  %d%% Inlier\nMAE      |  %d%% Inlier  "%(
      alpha1,
      alpha2)
ax[0].text(0.0,-0.13, temp_txt,
    horizontalalignment = 'left',
    verticalalignment = 'center',
    rotation = 'horizontal',
    fontsize = font_s,
    transform = ax[0].transAxes)

ax[1].set_title("True Dist.", fontsize=font_t+2)
im = ax[1].imshow(gt_dist, cmap="plasma")
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=font_s)
ax[1].axis('off')

for plot_idx in [2,3,4]:
    
    if plot_idx == 2:
        method_name = r'%d-bin OEDH $\rho_1$'%N_edhbins
        pred_depth = oedh_pred_depth1.cpu().numpy()
    elif plot_idx == 3:
        method_name = r'%d-bin PEDH $\rho_1$'%N_edhbins
        pred_depth = pedh_pred_depth1.cpu().numpy()
    elif plot_idx == 4:
        method_name = r'%d-bin DeePEDH $\rho_1$'%N_edhbins
        pred_depth = deepedh_pred_depth.cpu().numpy()
    

    ax[plot_idx].set_title(method_name, fontsize=font_t)
    # Clipping the output to ensure better comparision
    # ax[2].imshow(np.clip(pred_depth, depth.min(), depth.max()))
    im = ax[plot_idx].imshow(np.clip(pred_depth, gt_dist.min(), gt_dist.max()), cmap="plasma")
    ax[plot_idx].axis('off')
    divider = make_axes_locatable(ax[plot_idx])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=font_s)


    rmse_img, rmse_val = rmse(pred_depth, gt_dist)
    mae_img, mae_val = ame(pred_depth, gt_dist)
    pinl_mask1, pinl1 = p_inlier(pred_depth, gt_dist, alpha1)
    pinl_mask2, pinl2 = p_inlier(pred_depth, gt_dist, alpha2)

    temp_txt = "%.2f          |  %.2f  \n%.2f          |  %.2f"%(
        rmse_val,
        pinl1,
        mae_val,
        pinl2)
    ax[plot_idx].text(0.0,-0.13, temp_txt,
        horizontalalignment = 'left',
        verticalalignment = 'center',
        rotation = 'horizontal',
        fontsize = font_s,
        transform = ax[plot_idx].transAxes)
plt.suptitle(r'Distance Estimation Results for $\Phi_{sig} =$ %.2f, $\Phi_{sig} =$ %.2f'%(alpha_sig, alpha_bkg), fontsize = 32)
plt.show()
fig.savefig("DeePEDH_NYU_Result.png")
