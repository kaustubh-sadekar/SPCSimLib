from SPCSim.data_loaders.perpixel_loaders import PerPixelLoader

PixLdr = PerPixelLoader(num_dists = 4,
                        min_dist = 0,
                        max_dist = 1,
                        sig_bkg_list = [
                            [1,1],
                            [1,2]
                        ],
                        tmax = 100,
                        num_runs = 3,
                        device = "cuda")

# Generate the per pixel data
data = PixLdr.get_data()

# Output from PerPixelLoader is in form of a dictionary
# Use relevent keys to access specific typ0e of data
gt_dist = data["gt_dist"]


# Extract specific data modality
gt_dist = data["gt_dist"]
alpha_sig = data["alpha_sig"]
alpha_bkg = data["alpha_bkg"]

print("Absolute maximum distance :", PixLdr.dmax)
print("GT dist: ", gt_dist, gt_dist.shape)
print("Note the boradcasted values for illumination conditions for each run and each distance")
print("alpha_sig: ", alpha_sig, alpha_sig.shape)
print("alpha_bkg: ", alpha_bkg, alpha_bkg.shape)


# Accessing specific data point for given illumination and scene distance form the output data
# Access distance value, alpha_sig and alpha_bkg for the first set of SBR and first distance idx
print("Data for first SBR conditions and first distance value for second run")
ROW = PixLdr.get_row(sbr_idx =0, dist_idx=0)
RUN = 2

print("Dist = ", data["gt_dist"][ROW, RUN])
print("Alpha sig = ", data["alpha_sig"][ROW, RUN])
print("Alpha bkg = ", data["alpha_bkg"][ROW, RUN])