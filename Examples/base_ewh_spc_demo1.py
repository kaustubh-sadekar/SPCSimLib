from SPCSim.data_loaders.perpixel_loaders import PerPixelLoader
from SPCSim.data_loaders.transient_loaders import TransientGenerator
import matplotlib.pyplot as plt
from SPCSim.sensors.dtof import BaseEWHSPC
import torch
from SPCSim.utils.plot_utils import plot_transient, plot_ewh


# Simulating results for distance = 0.1*dmax
PixLdr = PerPixelLoader(min_dist = 0.4,
                        tmax = 100,
                        sig_bkg_list = [
                            [0.5,0.5]],
                        device = "cpu")

# Generate the per pixel data
data = PixLdr.get_data()


# Creating transient generator with laser time period of 100ns, FWHM 1 and with
# laser time period divided into 1000 equal time-bins
tr_gen = TransientGenerator(N_tbins = 1000, tmax = PixLdr.tmax, FWHM = 1)


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
N_ewhbins = 100
N_pulses = 5000

spc1 = BaseEWHSPC(
                Nr,
                Nc,
                N_pulses,
                device,
                N_tbins,
                N_ewhbins
               )

captured_data = spc1.capture(phi_bar)
ewh_data = captured_data["ewh"].cpu().numpy()
phi_bar = phi_bar.cpu()

ewh_bins_axis = torch.linspace(0,N_tbins-N_tbins//N_ewhbins,N_ewhbins)

fig, ax = plt.subplots(1,1,figsize=(8,4))
ROW, COL = [0,0]
plot_ewh(ax, ewh_bins_axis, ewh_data[ROW, COL,:], label = "EWH histogram", color = 'w')
plot_transient(ax, phi_bar[ROW, COL,:].numpy()*spc1.N_pulses, plt_type = '-r', label="True Transient")
ax.set_xlabel("Time (a.u.)")
ax.set_ylabel("Photon counts")
plt.legend()
fig.savefig("Temp.png")
