from SPCSim.data_loaders.perpixel_loaders import PerPixelLoader
from SPCSim.data_loaders.transient_loaders import TransientGenerator
from SPCSim.utils.plot_utils import plot_transient, plot_edh, plot_edh_traj
from SPCSim.sensors.dtof import HEDHBaseClass, PEDHOptimized
from SPCSim.postproc.edh_postproc import PostProcEDH
import matplotlib.pyplot as plt
import torch
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


N_edhbins = 2 # Number of EDH bins
N_pulses = 2000 # Number of laser pulses
N_tbins = 1000 # Number of time bins per laser cycle (B)
FWHM = 1 # in nano seconds
device = "cpu" # Select the CPU/GPU device
seed_val = 43

# Simulating results for distance = 0.2*MaxDistance, for laser time period = 100ns
# for average 1 signal photons per laser cycle and 1 background photons per laser cycle
PixLdr = PerPixelLoader(
                        min_dist = 0.2,
                        tmax = 100, # in nano seconds
                        sig_bkg_list = [[1,1]],
                        device = "cpu" # Choosing to run on CPU
                        )

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

# Set row and column of the pixel for which you want to track the CV trajectories
ROW, COL = [0,0]

# Initializing the HEDHBaseClass with desired sensor parameters
spc1 = HEDHBaseClass(PixLdr.Nr,
              PixLdr.Nc,
              N_pulses,
              device,
              N_tbins,
              N_edhbins,
              pix_r = ROW,
              pix_c = COL,
              seed=seed_val,
              step_params={
                  "k":1, # Step size gain
                  "step_vals": [1], # List of reducing step size values per binner
              })

# Capture method runs the HEDH binners for N_pulses based on the input true transient (phi_bar)
captured_data1 = spc1.capture(phi_bar)

# Extracting desired data components 
edh_data1 = captured_data1["edh"].cpu().numpy() # Contains the binner output for selected EDH class
gtedh_data1 = captured_data1["gtedh"].cpu().numpy() # Contains the ground truth EDH computed using the true transient
edh_list1 = captured_data1["traj"] # Contains the binner trajectories for pixel at location 
edh_list1 = np.array(edh_list1)


# Initializing the HEDHBaseClass with desired sensor parameters
spc2 = PEDHOptimized(PixLdr.Nr,
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

# Capture method runs the HEDH binners for N_pulses based on the input true transient (phi_bar)
captured_data2 = spc2.capture(phi_bar)

# Extracting desired data components 
edh_data2 = captured_data2["edh"].cpu().numpy() # Contains the binner output for selected EDH class
gtedh_data2 = captured_data2["gtedh"].cpu().numpy() # Contains the ground truth EDH computed using the true transient
edh_list2 = captured_data2["traj"] # Contains the binner trajectories for pixel at location 
edh_list2 = np.array(edh_list2)
phi_bar = phi_bar[0,0,:].cpu().numpy()

ewh_bins_axis = torch.linspace(0,N_tbins-1,N_tbins)

fig1, ax1 = plt.subplots(1,1, figsize=(6,6))
ax1.set_title(r'CV trajectories for different update strategies')
ax1.plot(edh_list1, label="Fixed stepping", alpha=0.5)
ax1.plot(edh_list2,'-r',label="Optimized stepping", alpha=0.6)
ax1.plot(phi_bar*N_pulses,ewh_bins_axis, '-k', label="True Transient")
ax1.axhline(gtedh_data2[0,0,1], color = '#097969', label = "True Median")
ax1.set_xlim(left = -0.5)
ax1.set_xlim(left = 0, right = 1500)
ax1.set_ylim(top = 520, bottom = 150)
ax1.legend(fontsize="11")
plt.gcf().set_dpi(400)
plt.tight_layout()
fig1.savefig("Temp_Fig3.png")
