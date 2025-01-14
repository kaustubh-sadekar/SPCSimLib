from SPCSim.data_loaders.transient_loaders import TransientGenerator
from SPCSim.utils.plot_utils import plot_transient
import matplotlib.pyplot as plt
import torch

# Creating transient generator with laser time period of 100ns, FWHM 1 and with
# laser time period divided into 1000 equal time-bins
tr_gen = TransientGenerator(N_tbins = 1000, tmax = 100, FWHM = 1)


device = "cuda"
dist_val = torch.tensor([[9.5]], device=device)
albedo = torch.tensor([[1.0]], device=device)
alpha_sig = torch.tensor([[1.0]], device=device)
alpha_bkg = torch.tensor([[2.0]], device=device)

# Using the get function to generate the transient
# for a given distance, albedo, intensity, and illumination condition
phi_bar = tr_gen.get_transient(dist_val,
                               albedo,
                               albedo,
                               alpha_sig,
                               alpha_bkg)


print("Shape of transient image: ",phi_bar.shape)

fig, ax = plt.subplots(1,1,figsize=(8,4))

plot_transient(ax, phi_bar.cpu().numpy()[0,0,:])
ax.set_title("Transient for dist = %.2f m, FWHM = %.2f ns"%(dist_val[0,0], tr_gen.FWHM))
fig.savefig("Temp.png")

print("Plot saved as Temp.png ...")
