from SPCSim.data_loaders.perpixel_loaders import PerPixelLoader
from SPCSim.data_loaders.transient_loaders import TransientGenerator
from SPCSim.utils.plot_utils import plot_transient, plot_ewh, plot_edh
import matplotlib.pyplot as plt
from SPCSim.sensors.dtof import RawSPC, BaseEWHSPC, BaseEDHSPC
import torch
import numpy as np


# Simulating results for distance = 0.3*MaxDistance, for laser time period = 100ns
# for average 0.1 signal photons per laser cycle and 0.2 background photons per laser cycle
PixLdr = PerPixelLoader(
                        min_dist = 0.3,
                        tmax = 100, # in nano seconds
                        sig_bkg_list = [[0.1,0.2]],
                        device = "cpu" # Choosing to run on CPU
                        )

# Generate the per pixel data
data = PixLdr.get_data()


# Creating transient generator with laser time period of 100ns, FWHM (pulse width) 5ns and with
# laser time period divided into 1000 equal time-bins
tr_gen = TransientGenerator(Nr = PixLdr.Nr, Nc = PixLdr.Nc, N_tbins = 1000, tmax = PixLdr.tmax, FWHM = 5)


# Using the get_transient function to generate the transient
# for a given distance, albedo, intensity, and illumination condition
phi_bar = tr_gen.get_transient(data["gt_dist"],
                               data["albedo"], # By default albedo is just a array of ones
                               data["albedo"],
                               data["alpha_sig"],
                               data["alpha_bkg"])

Nr, Nc, N_tbins = phi_bar.shape
device = PixLdr.device
seed_val = 43

# Simulating data for 500 laser cycles
N_pulses = 100

# Setting RawSPC to return 500 timestamps per pixel over the total exposure time.
# As RawSPC simulates single photon timestamp per laser cycle 
N_output_ts = N_pulses

spc1 = RawSPC(Nr,
              Nc,
              N_pulses,
              device,
              N_tbins,
              N_output_ts,
              seed=seed_val)

# Captured data contains timestamps (Nr x Nc x N_output_ts)
captured_data1 = spc1.capture(phi_bar)

# Accessing the timestamp data
raw_data = captured_data1["time_stamps"]


# Simulating data for 8-bin BaseEWHSPC sensor
N_ewhbins = 8
spc2 = BaseEWHSPC(
                Nr,
                Nc,
                N_pulses,
                device,
                N_tbins,
                N_ewhbins,
                seed=seed_val
               )

# Captured data contains equi-width histogram (Nr x Nc x N_ewhbins)
captured_data2 = spc2.capture(phi_bar)

# Accessing the equi-width histogram
ewh_data = captured_data2["ewh"]

# Simulating data for 8-bin BaseEDHSPC sensor
N_edhbins = N_ewhbins
spc3 = BaseEDHSPC(Nr,
              Nc,
              N_pulses,
              device,
              N_tbins,
              N_edhbins,
              seed=seed_val)

# Captured data contains equi-depth histogram (Nr x Nc x N_ewhbins)
captured_data3 = spc3.capture(phi_bar)

# Accessing the equi-depth histogram
oedh_data = captured_data3["oedh"]


# Plotting the results 
ROW, COL = [0,0]

phi_bar1 = phi_bar[ROW, COL, :].cpu().numpy()
ts = raw_data[ROW, COL, :].cpu().numpy().flatten()
ewh1 = ewh_data[ROW, COL, :].cpu().numpy()
edh1 = oedh_data[ROW, COL, :].cpu().numpy()

ewh_bins_axis = torch.linspace(0,N_tbins-N_tbins//N_ewhbins,N_ewhbins)
EDH_Height = (((data["alpha_sig"][ROW,COL]+data["alpha_bkg"][ROW,COL])*N_pulses/N_edhbins))


fig1, (ax1, ax2) = plt.subplots(1,2,figsize = (10,4))
plot_ewh(ax1, ewh_bins_axis, ewh1, label = "EWH histogram", color = 'w')
plot_transient(ax1, phi_bar1*50*spc1.N_pulses, plt_type = '-r', label="True Transient")
ax1.plot(ts, np.random.rand(ts.shape[-1])*0.001-1,'xk', label='Photon timestamps', linewidth=1)
csfont = {'fontname':'Times New Roman'}
ax1.set_title("Equi-width histogram (EWH)")
ax1.set_xlabel("Discretized time (a.u.)\n(a)")
ax1.set_ylabel("Photon counts")
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2.set_title("Equi-depth histogram (EDH)")
# ax2.bar(edh1[:-1], EDH_Height, width = edh_widths,color='w', alpha=0.5, edgecolor = 'black', align = 'edge', linewidth=1)
plot_edh(edh1,ax2,ymax = EDH_Height, colors_list=['k','k','k','k','k','k'])
plot_transient(ax2, phi_bar1*50*spc1.N_pulses, plt_type = '-r', label="True Transient")
ax2.plot(ts, np.random.rand(ts.shape[-1])*0.001-1,'xk', label='Photon timestamps', linewidth=1)
ax2.legend(frameon=False,fontsize="12",loc='upper right')
# ax2.set_ylim(top = EWH.max()*1.2)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlabel("Discretized time (a.u.)\n(b)")
ax2.set_ylabel("Photon counts")
plt.gcf().set_dpi(400)
# plt.tight_layout()
fig1.savefig("Temp_Fig2.png")
