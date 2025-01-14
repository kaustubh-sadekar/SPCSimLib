from SPCSim.data_loaders.perpixel_loaders import PerPixelLoader
from SPCSim.data_loaders.transient_loaders import TransientGenerator
from SPCSim.utils.plot_utils import plot_transient
import matplotlib.pyplot as plt
from SPCSim.sensors.dtof import RawSPC
import torch
import numpy as np


# Simulating results for distance = 0.1*dmax
PixLdr = PerPixelLoader(
                        num_dists=5,
                        min_dist = 0.2,
                        max_dist = 0.8,
                        tmax = 100,
                        sig_bkg_list = [
                            [0.2,0.2]],
                        num_runs=5,
                        device = "cpu")

# Generate the per pixel data
data = PixLdr.get_data()


# Creating transient generator with laser time period of 100ns, FWHM 1 and with
# laser time period divided into 1000 equal time-bins
tr_gen = TransientGenerator(Nr = PixLdr.Nr, Nc = PixLdr.Nc, N_tbins = 1000, tmax = PixLdr.tmax, FWHM = 2)


# Using the get function to generate the transient
# for a given distance, albedo, intensity, and illumination condition
phi_bar = tr_gen.get_transient(data["gt_dist"],
                               data["albedo"],
                               data["albedo"],
                               data["alpha_sig"],
                               data["alpha_bkg"])

Nr, Nc, N_tbins = phi_bar.shape
device = PixLdr.device
N_output_ts = 500
N_pulses = N_output_ts

spc1 = RawSPC(Nr,
              Nc,
              N_pulses,
              device,
              N_tbins,
              N_output_ts)

# Captured data contains timestamps (Nr x Nc x N_output_ts) and ewh (Nr x Nc x N_tbins)
captured_data = spc1.capture(phi_bar)

# Accessing the timestamp data
raw_data = captured_data["time_stamps"]

# Accessing the corresponding ewh
ewh_data = captured_data["ewh"]


# raw_data = raw_data.view(-1,raw_data.size()[0],raw_data.size()[1])

print("raw_data", raw_data.shape, raw_data.min(), raw_data.max())
fig, ax = plt.subplots(1,4,figsize=(20,4))
xaxis = torch.arange(0.5,1+N_tbins).to(torch.float)
print(xaxis.shape, xaxis)
sbr_idx = 0

# For first distance value and second run
ROW = PixLdr.get_row(sbr_idx =sbr_idx, dist_idx=0)
RUN = 2
hist,_ = torch.histogram(raw_data[ROW,RUN,:], xaxis)
hist2 = ewh_data[ROW,RUN,:]
plot_transient(ax[0], hist2.cpu().numpy(), plt_type = '-b', label="Captured EWH")
plot_transient(ax[0], hist.cpu().numpy(), plt_type = '--r', label="Timestamps histogram")
plot_transient(ax[0], phi_bar[ROW,RUN,:].cpu().numpy()*spc1.N_output_ts/np.mean(np.sum(phi_bar.cpu().numpy(), axis=-1)), plt_type = '-g', label="True Transient")
ax[0].set_xlabel('Bins')
ax[0].set_ylabel('Frequency')
ax[0].set_title('Histogram of raw data for \n' + r' Dist = %.2f m $(\Phi_{sig}, \Phi_{bkg})$ = (%.2f, %.2f)'%(data["gt_dist"][ROW, RUN], data["alpha_sig"][ROW, RUN], data["alpha_bkg"][ROW, RUN]))
ax[0].set_ylim(0, N_output_ts*0.2*(PixLdr.sig_bkg_list[sbr_idx][0] + PixLdr.sig_bkg_list[sbr_idx][1]))

# For second distance value and second run
ROW = PixLdr.get_row(sbr_idx =sbr_idx, dist_idx=1)
RUN = 2
hist,_ = torch.histogram(raw_data[ROW,RUN,:], xaxis)
hist2 = ewh_data[ROW,RUN,:]
plot_transient(ax[1], hist2.cpu().numpy(), plt_type = '-b', label="Captured EWH")
plot_transient(ax[1], hist.cpu().numpy(), plt_type = '--r', label="Timestamps histogram")
plot_transient(ax[1], phi_bar[ROW,RUN,:].cpu().numpy()*spc1.N_output_ts/np.mean(np.sum(phi_bar.cpu().numpy(), axis=-1)), plt_type = '-g', label="True Transient")
ax[1].set_xlabel('Bins')
ax[1].set_ylabel('Frequency')
ax[1].set_title('Histogram of raw data for \n' + r' Dist = %.2f m $(\Phi_{sig}, \Phi_{bkg})$ = (%.2f, %.2f)'%(data["gt_dist"][ROW, RUN], data["alpha_sig"][ROW, RUN], data["alpha_bkg"][ROW, RUN]))
ax[1].set_ylim(0, N_output_ts*0.2*(PixLdr.sig_bkg_list[sbr_idx][0] + PixLdr.sig_bkg_list[sbr_idx][1]))

# For third distance value and second run
ROW = PixLdr.get_row(sbr_idx =sbr_idx, dist_idx=2)
RUN = 2
hist,_ = torch.histogram(raw_data[ROW,RUN,:], xaxis)
hist2 = ewh_data[ROW,RUN,:]
plot_transient(ax[2], hist2.cpu().numpy(), plt_type = '-b', label="Captured EWH")
plot_transient(ax[2], hist.cpu().numpy(), plt_type = '--r', label="Timestamps histogram")
plot_transient(ax[2], phi_bar[ROW,RUN,:].cpu().numpy()*spc1.N_output_ts/np.mean(np.sum(phi_bar.cpu().numpy(), axis=-1)), plt_type = '-g', label="True Transient")
ax[2].set_xlabel('Bins')
ax[2].set_ylabel('Frequency')
ax[2].set_title('Histogram of raw data for \n' + r' Dist = %.2f m $(\Phi_{sig}, \Phi_{bkg})$ = (%.2f, %.2f)'%(data["gt_dist"][ROW, RUN], data["alpha_sig"][ROW, RUN], data["alpha_bkg"][ROW, RUN]))
ax[2].set_ylim(0, N_output_ts*0.2*(PixLdr.sig_bkg_list[sbr_idx][0] + PixLdr.sig_bkg_list[sbr_idx][1]))

# For fourth distance value and second run
ROW = PixLdr.get_row(sbr_idx =sbr_idx, dist_idx=3)
RUN = 2

hist,_ = torch.histogram(raw_data[ROW,RUN,:], xaxis)
hist2 = ewh_data[ROW,RUN,:]
plot_transient(ax[3], hist2.cpu().numpy(), plt_type = '-b', label="Captured EWH")
plot_transient(ax[3], hist.cpu().numpy(), plt_type = '--r', label="Timestamps histogram")
plot_transient(ax[3], phi_bar[ROW,RUN,:].cpu().numpy()*spc1.N_output_ts/np.mean(np.sum(phi_bar.cpu().numpy(), axis=-1)), plt_type = '-g', label="True Transient")
ax[3].set_xlabel('Bins')
ax[3].set_ylabel('Frequency')
ax[3].set_title('Histogram of raw data for \n' + r' Dist = %.2f m $(\Phi_{sig}, \Phi_{bkg})$ = (%.2f, %.2f)'%(data["gt_dist"][ROW, RUN], data["alpha_sig"][ROW, RUN], data["alpha_bkg"][ROW, RUN]))
ax[3].set_ylim(0, N_output_ts*0.2*(PixLdr.sig_bkg_list[sbr_idx][0] + PixLdr.sig_bkg_list[sbr_idx][1]))
plt.legend()
plt.savefig("Temp.png")
