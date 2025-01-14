import numpy as np
import cv2
import torch

class PerPixelLoader:
  r"""
  This class loads a custom RGB-D image with each row consisting of different
  distance values and avg signal-background photons and R columns where R is the
  number of runs to be captured for each pixel with a specific SBR and distance
  combination.
  
  .. note:: If A is the output transient matrix then A[i,j] gives readings for 
            ``dist_idx = i % len(SBR_list)``, ``SBR_Pair_idx = (i/num_dists)``, and ``Run_idx = j``.
  """
  def __init__(self,
               num_dists = 1,
               min_dist = 0,
               max_dist = 1,
               sig_bkg_list = [
                          [1,1]
                          ],
               tmax = 100, #time period in nano-seconds
               num_runs = 1,
               device = 'cpu',
               ):
    self.dmax = 3*1e8*tmax*1e-9/2 # scene distance in meters dmax = (c*tmax)/2
    self.num_runs = num_runs
    self.num_dists = num_dists
    self.min_dist = min_dist
    self.max_dist = max_dist
    self.sig_bkg_list = sig_bkg_list
    self.device = device
    self.tmax = tmax
    self.Nr = None
    self.Nc = None

  def get_data(self):
    r"""Method to generate the rgb-d data along with the average signal and background photon flux per cycle.

    Returns:
        data (dictionary): Dictionary containing the generated RGB, distance image, albedo, signal and bkg flux
    """
    
    dist_list = torch.tensor(np.linspace(self.min_dist, self.max_dist, num =self.num_dists).reshape(-1, 1))
    nr,nc = [len(self.sig_bkg_list)*self.num_dists,self.num_runs]
    alpha_sig = torch.zeros(nr, nc, 1)
    alpha_bkg = torch.zeros(nr, nc, 1)
    dist_f = torch.zeros(nr, nc, 1)

    for row in range(len(self.sig_bkg_list)):
      alpha_sig[row*self.num_dists:(row+1)*self.num_dists, :] = self.sig_bkg_list[row][0]
      alpha_bkg[row*self.num_dists:(row+1)*self.num_dists, :] = self.sig_bkg_list[row][1]
      dist_f[row*self.num_dists:(row+1)*self.num_dists, :] = dist_list.view(-1,1,1)

    albedo = torch.from_numpy(np.ones((nr,nc))).to(self.device)
    rgb = torch.from_numpy(np.ones((nr,nc,3))).to(self.device)
    gt_dist_factor = dist_f.view(nr,nc).to(self.device)
    gt_dist = gt_dist_factor*self.dmax

    self.Nr = nr
    self.Nc = nc

    data = {
        'rgb':rgb.to(self.device),
        'albedo':albedo.to(self.device),
        'gt_dist':gt_dist.to(self.device),
        'alpha_sig':alpha_sig.to(self.device),
        'alpha_bkg':alpha_bkg.to(self.device),
        'loader_id':"perpixel"
    }

    return data
  
  def get_row(self, sbr_idx=0, dist_idx=0):
    return sbr_idx*self.num_dists + dist_idx