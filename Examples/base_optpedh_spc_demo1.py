from SPCSim.data_loaders.perpixel_loaders import PerPixelLoader
from SPCSim.data_loaders.transient_loaders import TransientGenerator
from SPCSim.utils.plot_utils import plot_transient, plot_edh, plot_edh_traj
from SPCSim.sensors.dtof import PEDHOptimized
from SPCSim.postproc.edh_postproc import PostProcEDH
import matplotlib.pyplot as plt
import torch
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


N_edhbins = 32 # Number of EDH bins
N_pulses = 5000 # Number of laser pulses
N_tbins = 1000 # Number of time bins per laser cycle (B)
FWHM = 1 # in nano seconds
device = "cuda" # Select the CPU/GPU device
sbr_idx = 0 # Choose the index of the signal, background pair from sig_bkg_list to plot the output
dist_idx = 0 # Choose the index of the distance from equally spaced (num_dists) distances from min_dist to max_dist to plot the output
RUN = 0 # Choose the run number to plot the output
sims = 10 # Number of independent runs (i.e. simulations)

# Simulating results for distance values ranging from 0.1*N_tbins to 0.8*N_tbins
# for signal = [0.5, 1.0] and background = [0.5, 1.0, 2.5]
PixLdr = PerPixelLoader(min_dist = 0.7,
                        max_dist = 0.8,
                        num_dists= 10,
                        tmax = 100,
                        num_runs=sims,
                        sig_bkg_list = [
                            [0.5,0.5],
                            [0.5,1.0],
                            [0.5,2.5],
                            [1.0,0.5],
                            [1.0,1.0],
                            [1.0,2.5]],
                        device = device)

ROW = PixLdr.get_row(sbr_idx=sbr_idx, dist_idx=dist_idx)


# Generate the per pixel data
data = PixLdr.get_data()


# Creating transient generator with laser time period of tmax ns, FWHM and with
# laser time period divided into N_tbins equal time-bins
tr_gen = TransientGenerator(Nr = PixLdr.Nr, Nc = PixLdr.Nc, N_tbins = N_tbins, tmax = PixLdr.tmax, FWHM = FWHM)


# Using the get function to generate the transient
# for a given distance, albedo, intensity, and illumination condition
phi_bar = tr_gen.get_transient(data["gt_dist"], # NOTE: the true distance is in meters and depends on tmax
                               data["albedo"],
                               data["albedo"],
                               data["alpha_sig"],
                               data["alpha_bkg"])


# Initializing the HEDHBaseClass with desired sensor parameters
spc1 = PEDHOptimized(PixLdr.Nr,
              PixLdr.Nc,
              N_pulses,
              device,
              N_tbins,
              N_edhbins,
              pix_r = ROW,
              pix_c = RUN,
              step_params={
                  "k":3, # Step size gain
                  "decay":0,
                  "mtm":0.8,
                  "min_clip":0.02,
                  "switch_fraction":0.8,
                  "delta_mem": 0.95
              })

# Capture method runs the HEDH binners for N_pulses based on the input true transient (phi_bar)
captured_data = spc1.capture(phi_bar)

# Extracting desired data components 
edh_data = captured_data["edh"].cpu().numpy() # Contains the binner output for selected EDH class
gtedh_data = captured_data["gtedh"].cpu().numpy() # Contains the ground truth EDH computed using the true transient
ewh_data = captured_data["ewh"].cpu().numpy() # Contains the oracle EDH computed using the captured equi-width histogram
edh_list = captured_data["traj"] # Contains the binner trajectories for pixel at location 
edh_list = np.array(edh_list)

phi_bar = phi_bar.cpu().numpy() # Transferring the true transient to numpy array from torch tensor


# Creating instance of the class used to post process the EDH data to extract distance estimates
postproc = PostProcEDH(PixLdr.Nr,
              PixLdr.Nc,
              N_tbins,
              PixLdr.tmax,
              device)

# Pass the EDH output to edh2depth_t method as a torch tensor selected `device` and choose mode = 2 for narrowest bin distance estimate
# NOTE: use `dist_idx_` for distance estimates in bin location value and `dist_` for distance estimates in absolute meters
# NOTE: This method returns all torch.tensor values on the selected `device`.
bin_w_inv_, bin_idx_, dist_idx_, dist_ = postproc.edh2depth_t(torch.tensor(edh_data[:,:,1:-1], device=device), mode=2)

pred_dist = dist_.cpu().numpy()
gt_dist = data["gt_dist"].cpu().numpy()


fig, ax = plt.subplots(1,1, figsize=(10,4))
ymax = ((np.sum(ewh_data[ROW,RUN,:])/N_edhbins)).item()
plot_edh(edh_data[ROW,RUN,:],
         ax,
         ymax = ymax)
plot_edh(gtedh_data[ROW,RUN,:], ax,
         tr = phi_bar[ROW, RUN,:]*spc1.N_pulses,
         crop_window= tr_gen.FWHM*1.5*tr_gen.N_tbins*1.0/tr_gen.tmax,
         ymax = ymax, ls='--')
plot_transient(ax, ewh_data[ROW,RUN,:], plt_type='-b',label="EWH")

ax.set_title(r'Final EDH boundaries for $\Phi_{sig}$ = %.2f, $\Phi_{bkg}$ = %.2f, %d pulses'%(data["alpha_sig"][ROW,RUN], data["alpha_bkg"][ROW,RUN], spc1.N_pulses))
fig.savefig("Temp.png")

fig1, ax1 = plt.subplots(1,1, figsize=(8,8))
plot_edh_traj(ax1, edh_list, gtedh_data[ROW,RUN,1:-1], phi_bar[ROW, RUN,:]*spc1.N_pulses, ewh = ewh_data[ROW,RUN,:])
ax1.set_title(r'HEDH CV trajectories for $\Phi_{sig}$ = %.2f, $\Phi_{bkg}$ = %.2f, %d pulses'%(data["alpha_sig"][ROW,RUN], data["alpha_bkg"][ROW,RUN], spc1.N_pulses))
fig1.savefig("TempTraj.png")

fig2, ax2 = plt.subplots(1,2, figsize=(8,8))
im = ax2[0].imshow(gt_dist, cmap = 'jet')
ax2[0].axis('off')
ax2[0].set_title("True Dist")
divider = make_axes_locatable(ax2[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.ax.tick_params(length=0)
cbar.set_label('Distance (m)', rotation=270, labelpad=15)
im = ax2[1].imshow(pred_dist, cmap = 'jet')
ax2[1].axis('off')
ax2[1].set_title("Pred. Dist")
divider = make_axes_locatable(ax2[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.ax.tick_params(length=0)
cbar.set_label('Distance (m)', rotation=270, labelpad=15)

fig2.suptitle(r'Distance estimates for $\Phi_{sig}$ = %.2f, $\Phi_{bkg}$ = %.2f, %d pulses'%(data["alpha_sig"][ROW,RUN], data["alpha_bkg"][ROW,RUN], spc1.N_pulses))
fig2.savefig("DistanceOutput.png")
