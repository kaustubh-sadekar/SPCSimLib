from SPCSim.data_loaders.perpixel_loaders import PerPixelLoader
from SPCSim.data_loaders.transient_loaders import TransientGenerator
import matplotlib.pyplot as plt
from SPCSim.sensors.dtof import BaseEDHSPC
import torch
from SPCSim.utils.plot_utils import plot_transient, plot_edh


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
tr_gen = TransientGenerator(Nr = PixLdr.Nr, Nc = PixLdr.Nc, N_tbins = 1000, tmax = PixLdr.tmax, FWHM = 5)


# Using the get function to generate the transient
# for a given distance, albedo, intensity, and illumination condition
phi_bar = tr_gen.get_transient(
                                data["gt_dist"],
                                data["albedo"],
                                data["albedo"],
                                data["alpha_sig"],
                                data["alpha_bkg"])


# Setting the dimensions, device, number of EWH bins per pixel
# and number of laser pulses

Nr, Nc, N_tbins = phi_bar.shape
device = PixLdr.device
N_edhbins = 16
N_pulses = 5000

spc1 = BaseEDHSPC(Nr,
              Nc,
              N_pulses,
              device,
              N_tbins,
              N_edhbins)

captured_data = spc1.capture(phi_bar)
oedh_data = captured_data["oedh"]
gtedh_data = captured_data["gtedh"]

ewh_data = captured_data["ewh"]
fig, ax = plt.subplots(1,1, figsize=(12,4))

ROW, COL = [0,0]

ymax = ((torch.sum(ewh_data[ROW,COL,:])/N_edhbins)).item()

plot_edh(oedh_data[ROW,COL,:],
         ax,
         ymax = ymax)

plot_edh(gtedh_data[ROW,COL,:], ax,
         tr = phi_bar[ROW, COL,:].numpy()*spc1.N_pulses,
        #  crop_window= tr_gen.FWHM*1.5*tr_gen.N_tbins*1.0/tr_gen.tmax, # uncoment this line to zoom into peak
         ymax = ymax, ls='--')
fig.savefig("Temp.png")