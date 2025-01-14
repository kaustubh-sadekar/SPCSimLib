import numpy as np
import torch


class PostProcEWH:
    r"""Class for post processing the EWH data for distance estimation.
    """
    def __init__(self, Nr, Nc, N_tbins, tmax, device):
        r"""Set up the indexing arrays and common tensors for the class

        Args:
            Nr (int): Number of rows
            Nc (int): Number of columns
            device (str): Device `cpu` or `gpu`
            N_tbins (int): Number of time bins (frame)
            tmax (int)
        """
        self.Nr = Nr
        self.Nc = Nc
        self.N_tbins = N_tbins
        self.device = device
        self.tmax = tmax

    def ewh2depth_t(self, ewh):
        r""" Method to compute distance from equi-width histograms
        """

        factor = self.N_tbins//ewh.shape[-1]

        dist_idx = (torch.argmax(ewh, axis=2)+0.5)*(factor)

        tof = (dist_idx*1.0/self.N_tbins)*self.tmax

        dist = (tof*3e8*0.5*1e-9).to(self.device)
        
        return dist_idx, dist