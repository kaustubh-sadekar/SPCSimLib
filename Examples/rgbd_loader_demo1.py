import matplotlib.pyplot as plt
import pandas as pd
from SPCSim.data_loaders.rgbd_loaders import NYULoader1
from SPCSim.utils.plot_utils import plot_rgbd
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set correct path to NYUv2 dataset folder
csv_file_name = "/u/ksadekar/research/data/nyu_data/data/nyu2_train.csv"
root = "/u/ksadekar/research/data/nyu_data/"
df = pd.read_csv(csv_file_name, header=None)

# Initialize the NYUv2 data loader, set the index and choose the 
# correct paths to rgb and distance images
nyu_data = NYULoader1(256,256)
idx = 1
rgb_pth = root+df[0][idx]
dist_pth = root+df[1][idx]

# Using the get_data method to get rgbd data for specific rgbd frames path
data = nyu_data.get_data(rgb_pth, dist_pth, rgb_pth, rgb_pth)

fig, ax = plt.subplots(1,2, figsize=(8,4))
plot_rgbd(fig, 
          ax[0], 
          ax[1], 
          data["rgb"].cpu().numpy(), 
          data["gt_dist"].cpu().numpy(),
          show_cbar = True)

fig.savefig("Temp.png")
print("Figure saved as Temp.png ...")
