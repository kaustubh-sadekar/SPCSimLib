from SPCSim.data_loaders.perpixel_loaders import PerPixelLoader
from SPCSim.data_loaders.transient_loaders import TransientGenerator
from SPCSim.utils.plot_utils import plot_transient
import matplotlib.pyplot as plt

# Simulating results for multiple distance values ranging from 0.2xDmax to 0.8xDmax
PixLdr = PerPixelLoader(
                        num_dists = 5,
                        min_dist = 0.2,
                        max_dist = 0.8,
                        tmax = 100,
                        num_runs = 5,
                        sig_bkg_list=[
                            [1.0, 1.0],
                            [1.0, 10.0]],
                        device = "cuda")

# Generate the per pixel data
data = PixLdr.get_data()


# Creating transient generator with laser time period of 100ns, FWHM 1 and with
# laser time period divided into 1000 equal time-bins

# NOTE: unlike previous example, here we need to mention the
#       number of rows and cols (Nr, Nc)
tr_gen = TransientGenerator(Nr = PixLdr.Nr,
                            Nc = PixLdr.Nc,
                            N_tbins = 1000,
                            tmax = PixLdr.tmax,
                            FWHM = 2)


# Using the get function to generate the transient
# for a given distance, albedo, intensity, and illumination condition
phi_bar = tr_gen.get_transient(data["gt_dist"],
                               data["albedo"],
                               data["albedo"],
                               data["alpha_sig"],
                               data["alpha_bkg"])


print("Shape of transient image: ",phi_bar.shape)

fig, ax = plt.subplots(1,4,figsize=(24,3))

# Plotting first 4 distances with second SBR condition and second run
ROW = PixLdr.get_row(sbr_idx =1, dist_idx=0)
RUN = 2
plot_transient(ax[0], phi_bar.cpu().numpy()[ROW,RUN,:])
ax[0].set_title("Transient for dist = %.2f m, FWHM = %.2f ns"%(data["gt_dist"][ROW,RUN], tr_gen.FWHM))
ax[0].set_ylim(0, 0.2)

# Plotting first distance and second SBR
ROW = PixLdr.get_row(sbr_idx =1, dist_idx=1)
RUN = 2
plot_transient(ax[1], phi_bar.cpu().numpy()[ROW,RUN,:])
ax[1].set_title("Transient for dist = %.2f m, FWHM = %.2f ns"%(data["gt_dist"][ROW,RUN], tr_gen.FWHM))
ax[1].set_ylim(0, 0.2)

# Plotting second distance and first SBR
ROW = PixLdr.get_row(sbr_idx =1, dist_idx=2)
RUN = 2
plot_transient(ax[2], phi_bar.cpu().numpy()[ROW,RUN,:])
ax[2].set_title("Transient for dist = %.2f m, FWHM = %.2f ns"%(data["gt_dist"][ROW,RUN], tr_gen.FWHM))
ax[2].set_ylim(0, 0.2)

# Plotting second distance and second SBR
ROW = PixLdr.get_row(sbr_idx =1, dist_idx=3)
RUN = 2
plot_transient(ax[3], phi_bar.cpu().numpy()[ROW,RUN,:])
ax[3].set_title("Transient for dist = %.2f m, FWHM = %.2f ns"%(data["gt_dist"][ROW,RUN], tr_gen.FWHM))
ax[3].set_ylim(0, 0.2)

plt.plot()
print("Note the distance square fall off")
fig.savefig("Temp.png")