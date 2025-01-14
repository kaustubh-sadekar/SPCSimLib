from SPCSim.data_loaders.perpixel_loaders import PerPixelLoader
from SPCSim.data_loaders.transient_loaders import TransientGenerator
import matplotlib.pyplot as plt
from SPCSim.sensors.dtof import BaseEDHSPC, PEDHOptimized
from SPCSim.postproc.edh_postproc import PostProcEDH
import time
import torch
import numpy as np
from SPCSim.utils.plot_utils import plot_transient, plot_edh


seed_val = 43

# Simulating results for distance = 0.1*dmax
PixLdr = PerPixelLoader(min_dist = 0.32,
                        tmax = 100,
                        sig_bkg_list = [
                            [1,1]],
                        device = "cpu")

# Generate the per pixel data
data = PixLdr.get_data()


# Creating transient generator with laser time period of 100ns, FWHM 1 and with
# laser time period divided into 1000 equal time-bins
tr_gen = TransientGenerator(N_tbins = 1000, tmax = PixLdr.tmax, FWHM = 0.32)


# Using the get function to generate the transient
# for a given distance, albedo, intensity, and illumination condition
phi_bar = tr_gen.get_transient(data["gt_dist"],
                               data["albedo"],
                               data["albedo"],
                               data["alpha_sig"],
                               data["alpha_bkg"])


# Setting the dimensions, device, number of EWH bins per pixel
# and number of laser pulses

Nr, Nc, N_tbins = phi_bar.shape
device = PixLdr.device
N_edhbins = 32
N_pulses = 5000

# Set row and column of the pixel for which you want to track the CV trajectories
ROW, COL = [0,0]

spc1 = PEDHOptimized(PixLdr.Nr,
              PixLdr.Nc,
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


postproc = PostProcEDH(Nr, Nc, N_tbins, PixLdr.tmax, PixLdr.device)

captured_data = spc1.capture(phi_bar)
edh_data = captured_data["edh"]
print(edh_data.shape, edh_data)

oedh_data = captured_data["oedh"]
print(oedh_data.shape, oedh_data)

ewh_data = captured_data["ewh"]

oedh_rho0, _, _, oedh_pred_depth0 = postproc.edh2depth_t(oedh_data[:,:,1:-1], mode=0)
oedh_rho1, _, _, oedh_pred_depth1 = postproc.edh2depth_t(oedh_data[:,:,1:-1], mode=1)
oedh_bin_w_inv_, oedh_bin_idx_, _, oedh_pred_depth = postproc.edh2depth_t(oedh_data[:,:,1:-1], mode=2)

edh_rho0, _, _, edh_pred_depth0 = postproc.edh2depth_t(edh_data[:,:,1:-1], mode=0)
edh_rho1, _, _, edh_pred_depth1 = postproc.edh2depth_t(edh_data[:,:,1:-1], mode=1)
edh_bin_w_inv_, edh_bin_idx_, _, edh_pred_depth = postproc.edh2depth_t(edh_data[:,:,1:-1], mode=2)


fig, (ax, ax2) = plt.subplots(1,2, figsize=(12,4))


ymax = ((torch.sum(ewh_data[ROW,COL,:])/N_edhbins)).item()


sig, bkg = [data["alpha_sig"][ROW,COL].item(), data["alpha_bkg"][ROW, COL].item()]

edh_rho0_pix = edh_rho0[ROW, COL, :].cpu().numpy()
edh_rho1_pix = edh_rho1[ROW, COL, :].cpu().numpy()
edh_bin_w_inv_ = edh_bin_w_inv_[ROW, COL, :].cpu().numpy()
edh_bin_idx_ = edh_bin_idx_[ROW, COL, :].cpu().numpy()

oedh_rho0_pix = oedh_rho0[ROW, COL, :].cpu().numpy()
oedh_rho1_pix = oedh_rho1[ROW, COL, :].cpu().numpy()
oedh_bin_w_inv_ = oedh_bin_w_inv_[ROW, COL, :].cpu().numpy()
oedh_bin_idx_ = oedh_bin_idx_[ROW, COL, :].cpu().numpy()

tr1 = phi_bar[ROW, COL,:].cpu().numpy()
ax.set_title("(a) Oracle EDH")
plot_transient(ax, oedh_rho0_pix, label=r'$\rho_0$', plt_type='*-g')
plot_transient(ax, oedh_rho1_pix, label=r'$\rho_1$', plt_type='*-b')
plot_transient(ax, tr1*oedh_rho0_pix.max()/tr1.max(), label = r'True transient $\bar\Phi$', plt_type='-r')
# ax.step(oedh_bin_idx_,oedh_bin_w_inv_,"o:r", where="pre", label="True data points (step plot used in paper)")
ax.legend()
ax.set_xlim(left= max(np.argmax(tr1) - 10*tr_gen.smooth_sigma, 0), right = min(np.argmax(tr1) + 10*tr_gen.smooth_sigma, N_tbins))

ax2.set_title("(a) PEDH")
plot_transient(ax2, edh_rho0_pix, label=r'$\rho_0$', plt_type='*-g')
plot_transient(ax2, edh_rho1_pix, label=r'$\rho_1$', plt_type='*-b')
plot_transient(ax2, tr1*edh_rho0_pix.max()/tr1.max(), label = r'True transient $\bar\Phi$', plt_type='-r')
# ax.step(edh_bin_idx_,edh_bin_w_inv_,"o:r", where="pre", label="True data points (step plot used in paper)")
ax2.legend()

ax2.set_xlim(left= max(np.argmax(tr1) - 10*tr_gen.smooth_sigma, 0), right = min(np.argmax(tr1) + 10*tr_gen.smooth_sigma, N_tbins))

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize="12")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
fig.savefig("Temp_Fig4.png")
