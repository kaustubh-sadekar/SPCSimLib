import torch
import numpy as np

class TransientGenerator:
  def __init__(self, Nr = 1, Nc = 1, N_tbins = 1000, tmax = 100, FWHM = 1,  device = "cpu"):
    r"""Class to generate true transients without considering multi-path

    Args:
        Nr (int): Number of pixel rows in resized frame
        Nc (int): Number of pixel columns in resized frame
        N_tbins (int, optional): Number of discrete time bins dividing the laser time period. Defaults to 1000.
        tmax (int, optional): Laser time period in nano seconds. Defaults to 100.
        FWHM (int, optional): Laser full wave half maximum to decide the laser pulse width. Defaults to 1.
        device (str, optional): Choice of compute device. Defaults to 'cpu'.
    """

    self.N_tbins = N_tbins
    self.Nr = Nr
    self.Nc = Nc
    self.device = device
    self.dmax = 3*1e8*tmax*1e-9/2
    self.bin_size = tmax*1.0/N_tbins
    self.tmax = tmax
    self.FWHM = FWHM
    self.smooth_sigma = FWHM/(2.355*self.bin_size)
    self.smooth_window = (int(self.smooth_sigma*5)//2)*2 + 1
    x = torch.linspace(0,self.N_tbins-1,self.N_tbins)
    self.x = x.repeat(self.Nr,self.Nc,1).to(self.device)
    self.t = torch.zeros((self.Nr,self.Nc, self.N_tbins)).to(self.device)


  def get_signal_attenuation(self, albedo, dist):
    r"""Method to calculate attenuation factor from albedo information and dist information.

    Args:
        albedo (torch.tensor): Scene albedo image of dimension (Nr, Nc)
        dist (torch.tensor): Scene ground truth distance image of dimension (Nr, Nc)

    Returns:
        signal_attn (torch.tensor) : Attenuated signal
    """

    signal_attn = torch.divide(albedo*1.0,dist**2)
    self.temp_alpha = signal_attn.clone()
    signal_attn = signal_attn.reshape(signal_attn.shape[0],signal_attn.shape[1],1) #####
    return signal_attn

  def gt_shift_idx(self, gt_dist):
    r"""Method to compute the peak location in bins unit for given scene distance

    Args:
        gt_dist (torch.tensor): Scene ground truth distance image of dimension (Nr, Nc)

    Returns:
        shift_idx (torch.tensor) : Tensor consisting of integer values corresponding to the peak location.
    """

    shift_idx = torch.floor((gt_dist*1.0/self.dmax)*self.N_tbins).to(device = self.device,
                                                                      dtype = torch.int32)

    return shift_idx

  def get_shifted_laser_pulse_mesh(self, gt_dist):
    r"""Method to compute the time shifted gaussian pulse based on the true distance

    Args:
        gt_dist (torch.tensor): Scene ground truth distance image of dimension (Nr, Nc)

    Returns:
        tr (torch.tensor): Tensor of time shifted laser pulse for each pixel based on the true distance
    """
    mu = (self.gt_shift_idx(gt_dist)).long()
    mu = torch.reshape(mu, (self.Nr, self.Nc, 1)).to(device = self.device,
                                                    dtype = torch.int32)
    self.temp_range_bins = mu.clone()
    tr = (torch.exp(-((self.x - mu)**2)/(2*self.smooth_sigma**2)))/(self.smooth_sigma*np.sqrt(2*np.pi))
    sum_ = torch.sum(tr, 2)
    sum_ = torch.reshape(sum_, (self.Nr, self.Nc, 1))
    tr = torch.divide(tr, torch.sum(tr, axis=2, keepdims=True))
    return tr


  def get_transient(self, gt_dist, albedo, intensity, alpha_sig, alpha_bkg):
    r"""Method to add noise and attenuation to ideal transient

    Args:
        gt_dist (torch.tensor): Scene ground truth distance image of dimension (Nr, Nc)
        albedo (torch.tensor): Scene albedo image of dimension (Nr, Nc)
        intensity (torch.tensor): Scene intensity image of dimension (Nr, Nc)
        alpha_sig (float): Average signal photons per laser cycle
        alpha_bkg (float): Average background photons per laser cycle (not per bin)

    .. note:: Signal attenuation only depends on scene albedo and background flux depends on 
              the total scene intensity.
    

    Returns:
        r_t1 (torch.tensor): Time shifted attenuated laser pulse representing the photon density incident on the SPAD camera.
    """

    assert (gt_dist.shape[0] == self.Nr) and (gt_dist.shape[0] == self.Nr), \
      "Incorrect initialization of Nr and Nc for TransientGenerator \n must be %d, %d"%(gt_dist.shape[0], gt_dist.shape[1])

    r_t = self.get_shifted_laser_pulse_mesh(gt_dist)
    self.signal_attn = self.get_signal_attenuation(albedo/torch.mean(albedo), gt_dist)
    self.bkg_attn = (intensity/torch.mean(intensity)).reshape(albedo.shape[0],albedo.shape[1],1)
    self.k_signal = (self.signal_attn*alpha_sig/torch.mean(self.signal_attn)).to(self.device)
    self.k_bkg = (self.bkg_attn*alpha_bkg).to(self.device)
    r_t1 = torch.multiply(r_t, self.k_signal) + self.k_bkg/self.N_tbins
    return r_t1