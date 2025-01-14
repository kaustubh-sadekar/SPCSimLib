"""
##############
Sensors Module
##############


"""
import torch
import numpy as np
from tqdm import tqdm
import math
import sys

class BaseDtofSPC:
  r"""Base class for direct time-of-flight single-photon cameras.
  
  The capture method needs to be written for each of the inheriting classes.

  """
  def __init__(self, Nr, Nc, N_pulses, device, N_tbins, seed=0):
    r"""
    Args:
        Nr (int): Number of rows
        Nc (int): Number of columns
        N_pulses (int): Number of laser pulses
        device (str): Device `cpu` or `gpu`
        N_tbins (int): Number of time bins (frame)
    """
    
    self.Nr = Nr
    self.Nc = Nc
    self.N_pulses = N_pulses
    self.device = device
    self.N_tbins = N_tbins
    self.seed = seed
    if self.seed:
        torch.manual_seed(self.seed)
        # torch.use_deterministic_algorithms(True)
    self.ts_vec = torch.zeros((Nr,Nc,2*N_tbins),device=self.device)
    self.rr,self.cc  = torch.meshgrid(torch.linspace(0,self.Nr-1,self.Nr),
                                      torch.linspace(0,self.Nc-1,self.Nc), indexing='ij')
    self.rr = self.rr.to(device = self.device,
                                        dtype = torch.long)
    self.cc = self.cc.to(device = self.device,
                                        dtype = torch.long)
    
  def sim_poisson_process(self, phi_bar):    
    r""" Method to simulate detected photons from average incident photon flux.

    .. note:: The sim_poisson_process method of BaseDtofSPC takes input as phi_bar (the probability of detecting photons in each bin) and 
          applies the following operation ``hist = torch.poisson(phi_bar).to(device = self.device, dtype = torch.bool)*1``. It is important to note that
          once the per bin probability of detecting a photon increases above 1 or 1.3, almost all bins detect a photon hence the output vector `hist` is
          always a vector of all ones. Hence increasing the total photon flux above 1.3 photons per bin results in the same values of `hist` no mater how
          high the flux is. This is important to note when performing experiments for high-photon flux scenarios.
    
    """
    hist = torch.poisson(phi_bar) # This line does not perform sum normalization before passing phi_bar to torch.poisson sampling.
      # Performing a sum normalization ensures the total probability = 1 but we deal with per bin probabilities.

    hist = hist.to(device = self.device,
                    dtype = torch.bool)*1
    self.ts_vec[:,:,:self.N_tbins] = 0
    self.ts_vec[:,:,self.N_tbins:] = hist
    
    return (self.ts_vec[:,:,self.N_tbins:]*1).to(device = self.device,
                    dtype = torch.bool)*1

  def capture(self, phi_bar):
    r"""Method needs to be implemented for inheriting class

    Ensure the output to be a dictionary for future compatibility. Check inheriting classes for examples.

    Args:
        phi_bar (torch.tensor): Average incident photon flux tensor of shape (Nr, Nc, N_tbins)

    Raises:
        NotImplementedError: This is a base class method which needs to be implemented for each inheriting class.
    """
    raise NotImplementedError


class RawSPC(BaseDtofSPC):
  r""" This SPC class simulates N_output_ts photon time stamp measurements for NrxNc pixels.

  Either use the class with PerPixelLoader to simulate phi_bar data or pass transient
  from a dataset and capture all the timestamps for N_pulses laser cycles.

  .. note:: For this class if the average signal+bkg photons per laser cycle exceed 1 then there is a high change that 
          the we will run out of the `N_output_ts` timestamps even before we scan all the histogram bins.
  """
  def __init__(self,Nr, Nc, N_pulses, device, N_tbins, N_output_ts, seed=0):
    r"""
    Args:
        Nr (int): Number of rows
        Nc (int): Number of columns
        N_pulses (int): Number of laser pulses
        device (int): Device `cpu` or `gpu`
        N_tbins (int): Number of time bins (frame)
        N_output_ts (int): Number of time stamps to be generated per pixel.
    """
    BaseDtofSPC.__init__(self,Nr, Nc, N_pulses, device, N_tbins, seed=seed)
    # Note: timestamp index values are starting from 1 till n_tbins
    self.time_idx = (torch.arange(self.N_tbins)+1).to(device = self.device,
                                                      dtype = torch.long)
    self.N_output_ts = N_output_ts

  def capture(self, phi_bar):
    r"""Method to generate SPC data for average incident photon flux (phi_bar) for 
    given number of laser cycles and other SPC intrinsic parameters.

    Args:
        phi_bar (torch.tensor): Average incident photon flux tensor of shape (Nr, Nc, N_tbins)

    Returns:
        captured_data (dictionary): Dictionary containing raw photon timestamps (``time_stamps``) 
                                    and corresponding EW histogram (``ewh``) tensor
    """
    time_stamps = torch.zeros((self.Nr, self.Nc, self.N_output_ts))

    ewh = 0
    for cycles in tqdm(range(self.N_pulses)):
      ewh += self.sim_poisson_process(phi_bar)

    for row in range(self.Nr):
      for col in range(self.Nc):
        current_series = ewh[row, col, :]
        start_idx = 0

        for k in range(current_series.shape[-1]):
          if start_idx+current_series[k] > (self.N_output_ts):
              time_stamps[row, col, start_idx:self.N_output_ts] = k+0.5
              break
          else:
            time_stamps[row, col, start_idx:start_idx+current_series[k]] = k+0.5
          start_idx = start_idx+current_series[k]
        time_stamps[row, col,:] = time_stamps[row, col,torch.randperm(time_stamps.size()[-1])]
    
    captured_data = {
      "time_stamps": time_stamps,
      "ewh": ewh,
    }

    return captured_data


class BaseEWHSPC(BaseDtofSPC):
  r""" This SPC class simulates photon measurements and captures them in form of equi-width (EW) histograms.

  EW histograms divide the laser time period into bins of equal widths and store the count of photon timestamps 
  incident in respective bin. In idea scenarios the bin with highest counts is most likely to contain the signal peak.

  """
  def __init__(self,Nr, Nc, N_pulses, device, N_tbins, N_ewhbins, seed=0):
    BaseDtofSPC.__init__(self,Nr, Nc, N_pulses, device, N_tbins, seed=seed)
    r"""
    Args:
        Nr (int): Number of rows
        Nc (int): Number of columns
        N_pulses (int): Number of laser pulses
        device (int): Device `cpu` or `gpu`
        N_tbins (int): Number of time bins (frame)
        N_ewhbins (int): Number of equi-width histogram bins.
    """
    self.N_ewhbins = N_ewhbins

  def capture(self, phi_bar):
    r"""Method to generate SPC data for average incident photon flux (phi_bar) for 
    given number of laser cycles and other SPC intrinsic parameters.

    Args:
        phi_bar (torch.tensor): Average incident photon flux tensor of shape (Nr, Nc, N_tbins)

    Returns:
        captured_data (dictionary): Dictionary containing EW histogram tensor
    """
    hist = 0
    # NOTE: This step can be optimized by computing poisson process measurements for phi_bar*N_pulses instead of for loop
    for cycles in tqdm(range(self.N_pulses)):
      hist += self.sim_poisson_process(phi_bar) 
    
    factor = self.N_tbins//self.N_ewhbins
    ewh_data = torch.zeros((self.Nr, self.Nc, self.N_ewhbins), dtype=torch.int32, device=self.device)
    
    # Iterating over each EWH bin and counting the total photon counts
    for ewh_idx in range(self.N_ewhbins):
      ewh_data[:,:,ewh_idx] = torch.sum(hist[:,:,ewh_idx*factor:(ewh_idx+1)*factor],-1,keepdims=False)
    
    captured_data = {
      "ewh": ewh_data
    }

    return captured_data


class BaseEDHSPC(BaseDtofSPC):
  r""" This SPC class simulates photon measurements and captures them in form of equi-depth (ED) histograms.
  
  Unlike EWHSPCs the EDHSPCs divide the laser time period such that the (depth) total counts per bin is equal hence ED histogram bins 
  are mostly unequal in width. 

  .. note:: The term `depth` in equi-depth histograms refers to the count/height of the bin indicating that ED histogram bins have equal height. The term
            `depth` is not supposed to be confused with the scene depth. Hence in the code `distance` is used instead of `depth` to indicate how far objects are in the scene.
  """
  def __init__(self,Nr, Nc, N_pulses, device, N_tbins, N_edhbins, seed=0):
    r"""
    Args:
        Nr (int): Number of rows
        Nc (int): Number of columns
        N_pulses (int): Number of laser pulses
        device (int): Device `cpu` or `gpu`
        N_tbins (int): Number of time bins (frame)
        N_edhbins (int): Number of equi-depth histogram bins.
    """

    BaseDtofSPC.__init__(self,Nr, Nc, N_pulses, device, N_tbins, seed=seed)
    self.N_edhbins = N_edhbins
  
  def get_ts_from_hist(self, hist):
    r"""Method to convert one-hot encoded photon detection vectors to photon time stamp vectors

    Args:
        hist (torch.tensor): Input one-hot encoded photon detection cube where hist[n,m,k] = 1 if a photon is detected in the 
        kth time bin for pixel in the nth row and mth column

    Returns:
        ts (torch.tensor): Time stamp tensor of the same size as hist. ts[n,m,k] = k if hist[n,m,k] = 1 else =0
        hist (torch.tensor): Input one-hot encoded photon detection cube.
    """
    hist = hist.to(torch.long)
    ts = ((hist>0)*1)*self.time_idx
    return ts, hist

  def ewh2edh(self, ewh_data):
    r""" Method to compute equi-depth histogram from equi-width histogram.

    Args:
        ewh_data (torch.tensor): Equi-width histogram tensor of shape (Nr, Nc, N_tbins)

    Returns:
        edh_bins (torch.tensor): Oracle equi-depth histogram tensor of shape (Nr, Nc, N_tbins)
    """

    assert ewh_data.shape[-1] == self.N_tbins,"For this code version the number of EW histogram bins must be equal to N_tbins"

    tr_img = ewh_data + torch.max(ewh_data)*0.0000000001/self.N_tbins
    n_edh = self.N_edhbins
    r,c,bins = tr_img.shape
    tr_cs = torch.cumsum(tr_img, axis=2)
    tr_sum = torch.sum(tr_img, axis=2).reshape(r,c,1)
    edh_bins = torch.zeros((r,c, n_edh-1))

    for idx in range(edh_bins.shape[2]):
      e1_ori = tr_cs - tr_sum*(idx+1.0)/n_edh
      e1 = e1_ori*1.0
      e2 = e1_ori*1.0
      e1[e1_ori > 0] = -1000000000000.0
      e2[e1_ori < 0] = 1000000000000.0

      neg_max_val_, neg_max_idx_ = torch.max(e1, axis=-1)
      pos_min_val_, pos_min_idx_ = torch.min(e2, axis=-1)
      k1 = 1# pos_min_idx_ - neg_max_idx_
      a1 = torch.abs(neg_max_val_)
      b1 = pos_min_val_
      edh_bins[:,:,idx] = (neg_max_idx_ + a1*k1*1.0/(a1+b1+0.00000000000001))

    return edh_bins+1

  def capture(self, phi_bar):
    r"""Method to generate SPC data for average incident photon flux (phi_bar) for 
    given number of laser cycles and other SPC intrinsic parameters.

    Args:
        phi_bar (torch.tensor): Average incident photon flux tensor of shape (Nr, Nc, N_tbins)

    Returns:
        captured_data (dictionary): Dictionary containing ED histogram for detected photons and 
        the average incident photons flux and corresponding EW histogram tensor
    """
    
    hist = 0
    # NOTE: This step can be optimized by computing poisson process measurements for phi_bar*N_pulses instead of for loop
    # But the for loop will enable additional features in future versions.
    for cycles in tqdm(range(self.N_pulses)):
      hist += self.sim_poisson_process(phi_bar)
    
    oedh_bins = self.ewh2edh(hist)
    oedh_bins = self.add_extreme_boundaries(oedh_bins)
    
    gt_edh_bins = self.ewh2edh(phi_bar)
    gt_edh_bins = self.add_extreme_boundaries(gt_edh_bins)
    
    captured_data = {
      "oedh": oedh_bins, # ED histogram generated from measured EW histogram
      "gtedh": gt_edh_bins, # ED histogram generated from average incident photon flux (phi_bar)
      "ewh": hist
    }
    return captured_data
  
  def add_extreme_boundaries(self, edh):
    return torch.cat(((edh[:,:,0]*0).unsqueeze(-1), edh, (edh[:,:,0]*0+self.N_tbins).unsqueeze(-1)), axis=-1).to(self.device)


class HEDHBaseClass(BaseEDHSPC):
  r"""Base class for hierarchical EDH
  """
  def __init__(self, 
               Nr, 
               Nc,  
               N_pulses, 
               device, 
               N_tbins,
               N_edhbins,
               seed = 0, 
               save_traj = True, 
               pix_r = 0, 
               pix_c = 0, 
               step_params = {}):
    r"""Initialize SPC parameters

    Args:
        Nr (int): Number of rows
        Nc (int): Number of columns
        N_pulses (int): Number of laser pulses
        device (str): Device `cpu` or `gpu`
        N_tbins (int): Number of time bins (frame)
        N_edhbins (unt): Number of EDH bins
        seed (int, optional): Choose the random seed. Defaults to 0.
        save_traj (bool, optional): Set the flag to save binner trajectories. Defaults to True.
        pix_r (int, optional): Choose row number of the pixel to save trajectory. Defaults to 0.
        pix_c (int, optional): Choose column number of the pixel to save trajectory. Defaults to 0.
        step_params (dict, optional): Dictionary to pass different stepping parameters. Defaults to {}.
    """    
    BaseEDHSPC.__init__(self,Nr, Nc, N_pulses, device, N_tbins, N_edhbins, seed=seed)
    
    if not(step_params):
      # Set the default stepping params
      self.k = 1
      self.step_vals = [1]
    else:
      self.k = step_params["k"]
      self.step_vals = step_params["step_vals"]
    
    self.save_traj = save_traj
    self.N_levels = int(math.log2(self.N_edhbins))
    self.temp_delta = 0
    self.pix_r = pix_r
    self.pix_c = pix_c
    self.set_idx_lists()
    self.init_edh_params()    
    

  def set_idx_lists(self):
    r"""Method to set the binner indices for further operations.
    Reference image for 16-bin EDH

    .. image:: media/SPCSim_bins.png

    .. note:: For 16-bin HEDH, we track 15 boundaries and two extra boundaries on index 0 and -1 are the extreme boundaries.
              Since we do not perform any updates to the boundaries we do not pass the extreme boundaries for any edh update step.
    """
    if self.N_edhbins == 2:
      self.idx_list = [1]
      self.clip_left = [0]
      self.clip_right = [-1]
      self.level_order_list = [[1]]

    elif self.N_edhbins == 4:
      self.idx_list = [1,2,3]
      self.clip_left = [0,0,2]
      self.clip_right = [2,-1,-1]
      self.level_order_list = [[2],[1,3]]
    
    elif self.N_edhbins == 8:
      self.idx_list = [1,2,3,4,5,6,7]
      self.clip_left = [0,0,2,0,4,4,6]
      self.clip_right = [2,4,4,-1,6,-1,-1]
      self.level_order_list = [[4],[2,6],[1,3,5,7]]
    
    elif self.N_edhbins == 16:
      self.idx_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
      self.clip_left = [0,0,2,0,4,4,6,0,8,8,10,8,12,12,14]
      self.clip_right = [2,4,4,8,6,8,8,-1,10,12,12,-1,14,-1,-1]
      self.level_order_list = [[8],[4,12],[2,6,10,14],[1,3,5,7,9,11,13,15]]
    
    elif self.N_edhbins == 32:
      self.idx_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
      self.clip_left = [0,0,2,0,4,4,6,0,8,8,10,8,12,12,14,0,16,16,18,16,20,20,22,16,24,24,26,24,28,28,30]
      self.clip_right = [2,4,4,8,6,8,8,16,10,12,12,16,14,16,16,-1, 18,20,20,24,22,24,24,-1,26,28,28,-1,30,-1,-1]
      self.level_order_list = [[16],[8,24],[4,12,20,28],[2,6,10,14,18,22,26,30],[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]]
    
    else:
      sys.exit("HEDHBaseClass and inheriting classes only supports one of the following N_edhbins [2, 4, 8, 16, 32]")
    

    assert (len(self.idx_list) == len(self.clip_left) == len(self.clip_right)), [len(self.idx_list), len(self.clip_left), len(self.clip_right)]


  def init_edh_params(self):
    r"""This method is called in the constructor and ensures initializing all the necessary common tensors that will be updated later.
    Initializing the tensors in the constructor eliminates the need to construct them at each laser cycle hence reduces simulation time.
    """
    self.e1 = torch.zeros((self.Nr, self.Nc, self.N_edhbins-1)).to(torch.float).to(self.device)
    self.edh_bins = torch.zeros((self.Nr, self.Nc, self.N_edhbins+1)).to(torch.float).to(self.device)
    self.edh_bins[:,:,-1] = self.N_tbins
    self.pl = torch.zeros_like(self.e1).to(self.device) 
    self.pe = torch.zeros_like(self.e1).to(self.device)
    self.prev_step = torch.zeros_like(self.e1).to(self.device)
    self.prev_err = torch.zeros_like(self.e1).to(self.device)
    self.delu = torch.zeros_like(self.e1).to(self.device)
    self.e = self.e1.to(torch.long).to(self.device)
    self.cy_cnt = 0
    self.kp = 0
    self.delta = 0
    self.eps = self.N_tbins*10e-8

    self.ch = torch.arange(self.N_edhbins-1).reshape(1, 1, self.N_edhbins-1)+1
    self.ch = torch.tile(self.ch, (self.Nr, self.Nc,1))
    self.a = (self.ch*1.0/(self.N_edhbins)).reshape(self.Nr, self.Nc, self.N_edhbins-1).to(self.device)
    self.b = ((self.N_edhbins - self.ch)*1.0/(self.N_edhbins)).reshape(self.Nr, self.Nc, self.N_edhbins-1).to(self.device)

    # Note: index values are starting from 1 till n_tbins
    self.time_idx = (torch.arange(self.N_tbins)+1).to(device = self.device,
                                                      dtype = torch.long)

    self.rr,self.cc  = torch.meshgrid(torch.linspace(0,self.Nr-1,self.Nr),
                                          torch.linspace(0,self.Nc-1,self.Nc), indexing='ij')
    self.rr = self.rr.to(device = self.device,
                                        dtype = torch.long)
    self.cc = self.cc.to(device = self.device,
                                        dtype = torch.long)
    self.rr2 = self.rr.reshape(self.Nr, self.Nc,1)
    self.cc2 = self.cc.reshape(self.Nr, self.Nc,1)
    self.idx = torch.arange(self.N_edhbins-1)
    self.i = self.idx.repeat(self.Nr,self.Nc,1)
    self.reset_edh()
    self.set_decay_schedule()
    
    self.delta_mask = torch.zeros((self.Nr, self.Nc, self.N_edhbins-1)).to(self.device)

  def reset_edh(self):
    r"""This method is called once in the constructor to decide the initialization scheme of the binners at the first laser cycle. 

    .. note:: Valid-mask based initialization for binners at later stages is handled by :func:`~dtof.HEDHBaseClass.update_delta_mask`

    """
    temp = (((torch.arange(self.N_edhbins-1, dtype=torch.float)+1.0)*1.0/self.N_edhbins)*self.N_tbins).to(torch.long) - 1
    self.e[self.rr2,self.cc2,self.i] = temp.to(self.device)
    self.e1 = self.e.to(torch.float)

  def update_pa_pb_kp(self, hist, ts):
    r"""Method to compute the early and late photon streams for current control value of each binner in a vectorized form.

    Refer the following figure from [1] to understand how the binner hardware splits the photon stream into early and late photons.

    .. image:: https://static.wixstatic.com/media/bcd6ad_c4fbe0e7d57b47beafac0b091f502e97~mv2.png/v1/fill/w_600,h_491,al_c,q_85,usm_0.66_1.00_0.01,enc_avif,quality_auto/bcd6ad_c4fbe0e7d57b47beafac0b091f502e97~mv2.png
    

    Args:
        hist (torch.tensor): Input one-hot encoded photon detection cube.
        ts (torch.tensor): Time stamp tensor of the same size as hist. ts[n,m,k] = k if hist[n,m,k] = 1 else =0
    """

    # time stamps
    self.kp = self.kp*0
    self.pl = self.pl*0
    self.pe = self.pe*0

    self.kp = torch.sum(hist, axis=2).reshape(self.Nr, self.Nc, 1).to(self.device)

    for k in range(self.N_edhbins-1):
      aidx = self.clip_right[k]
      bidx = self.clip_left[k]

      pa_mask = (ts > self.e1[:,:,k].clone().reshape(self.Nr, self.Nc, 1).to(self.device))*(ts < self.edh_bins[:,:,aidx].clone().reshape(self.Nr, self.Nc, 1).to(self.device))
      pb_mask = (ts > self.edh_bins[:,:,bidx].clone().reshape(self.Nr, self.Nc, 1).to(self.device))*(ts < self.e1[:,:,k].clone().reshape(self.Nr, self.Nc, 1).to(self.device))

      # NOTE: This implementation can also handle cases when more than one photons are detected
      self.pl[:,:,k] = torch.sum(pa_mask*hist, axis=2)
      self.pe[:,:,k] = torch.sum(pb_mask*hist, axis=2)

  def update_delta(self):
    r"""Method to update the difference between early photons (``pe``) and late photons (``pl``). The `delta` value is used to compute the binner update step.
    
    For a basic binner, the sign of delta is used to compute the binner update step as ``step[k+1] = sign(delta[k])``. 
    """
    self.delta = (self.pl - self.pe).to(self.device)*self.delta_mask

  def apply_edh_step(self):
    r"""Method to apply the computed step size to update the control values of respective binners.

    For the HEDHSPCs it also ensures that the child boundaries do not cross over the parent boundaries.
    """

    for idx_per_level in self.level_order_list:
      for i_ in idx_per_level:
        # print(idx_per_level,"i = ",i_)
        new_val = self.e1[:,:,i_-1] + self.prev_step[:,:,i_-1]
        aidx = self.clip_left[i_-1]
        bidx = self.clip_right[i_-1]
        mask = (new_val > self.edh_bins[:,:,aidx])*(new_val<self.edh_bins[:,:,bidx])
        self.e1[:,:,i_-1] = self.e1[:,:,i_-1] + self.prev_step[:,:,i_-1]*mask

    # self.e1 = self.e1 + self.prev_step
    self.e = self.e1.to(device=self.device, dtype = torch.long)
  
  def set_decay_schedule(self):
    r""" Method to set the stepping schedule for the binners. 
    
    The ```decay_schedule``` contains a list of decay values for each laser cycle. Hence its length is equal to the number of laser cycles. 
    A simple example of could be to overwrite this method with the following definition

    .. code:: python

        def set_decay_schedule(self):
          self.decay_schedule = np.ones(self.N_pulses).tolist()

    """
    self.decay_schedule = []

    for i in range(self.N_levels):
      for step_size in self.step_vals:
        for k in range(int(self.N_pulses/(self.N_levels*len(self.step_vals)))):
          self.decay_schedule.append(step_size)
    
    print(len(self.decay_schedule))
    assert len(self.decay_schedule) == self.N_pulses, "Ensure N_pulses is a multiple of %d to distribute the exposure time equally between all binners"%(self.N_levels*len(self.step_vals))
  
  def update_delta_mask(self):
    r""" Method to apply custom temporal masks to activate or deactivate
    specific binners.

    Example to ensure all boundaries are activated for all the laser cycles
    
    .. code-block:: python

      def update_delta_mask(self):
        self.delta_mask = self.delta_mask*0 + 1
    """
    
    level = self.cy_cnt//(self.N_pulses//self.N_levels)
    
    self.delta_mask = self.delta_mask*0
    
    for i in self.level_order_list[level]:
      self.delta_mask[:,:,i-1] = 1
    
      if level and ((self.cy_cnt*1.0)%int(self.N_pulses//self.N_levels) == 0):
        aidx = self.clip_left[i-1]
        bidx = self.clip_right[i-1]
        self.e1[:,:,i-1] = torch.bitwise_right_shift(self.edh_bins[:,:,aidx].clone().to(torch.int)+self.edh_bins[:,:,bidx].clone().to(torch.int),1)

  def update_edh(self, hist):
    r"""Method to compute and apply the update step to the binners

    This method obtains the decay value for the specific cycle (cy_cnt), updates the delta_mask to choose valid binners to update at (cy_cnt),
    generates timestamp vectors from photon detection vectors, uses the timestamp vectors to compute before and after photons for each binner, 
    compute the delta (late photons - early photons)/total photons, computer the update step for each binner and apply the update step to corresponding binners.
    """

    self.decay = self.decay_schedule[self.cy_cnt]

    self.update_delta_mask()

    ts, hist = self.get_ts_from_hist(hist)
    self.update_pa_pb_kp(hist, ts)
    self.update_delta()

    self.delta = torch.sign(self.delta)

    new_step = (self.delta*self.k).to(self.device)
    
    self.prev_step = new_step*self.decay
    
    self.apply_edh_step()

    self.edh_bins[:,:,1:-1] = self.e1*1.0

    self.cy_cnt+=1

    return self.e1 
  

  def capture(self, phi_bar):
    r"""Method to generate SPC data for average incident photon flux (phi_bar) for 
    given number of laser cycles and other SPC intrinsic parameters.

    Args:
        phi_bar (torch.tensor): Average incident photon flux tensor of shape (Nr, Nc, N_tbins)

    Returns:
        captured_data (dictionary): Dictionary containing ED histogram for detected photons, the oracle and ground truth ED histograms, corresponding EW histogram tensor and 
        list of binner control values tracked by binner at each laser cycle for a pixel at ``(self.pix_r, self.pix_c)``. 
    """

    self.reset_edh()
    traj = []
    ewh_img = 0
    for i in tqdm(range(self.N_pulses)):
      hist = self.sim_poisson_process(phi_bar)
      ewh_img+=hist
      edh_img = self.update_edh(hist)
      if self.save_traj:
        traj.append(edh_img[self.pix_r,self.pix_c,:].cpu().tolist())
    

    edh_bins = self.e1
    edh_bins = self.add_extreme_boundaries(edh_bins)

    oedh_bins = self.ewh2edh(ewh_img)
    oedh_bins = self.add_extreme_boundaries(oedh_bins)

    gt_edh_bins = self.ewh2edh(phi_bar)
    gt_edh_bins = self.add_extreme_boundaries(gt_edh_bins)

    captured_data = {
        'edh':edh_bins,
        'oedh':oedh_bins,
        'gtedh':gt_edh_bins,
        'ewh':ewh_img,
        'traj':traj,
    }
    return captured_data



class PEDHBaseClass(BaseEDHSPC):
  def __init__(self, 
               Nr, 
               Nc,  
               N_pulses, 
               device, 
               N_tbins,
               N_edhbins,
               seed = 0, 
               save_traj = True, 
               pix_r = 0, 
               pix_c = 0, 
               step_params = {}):
    r"""Initialize SPC parameters

    Args:
        Nr (int): Number of rows
        Nc (int): Number of columns
        N_pulses (int): Number of laser pulses
        device (str): Device `cpu` or `gpu`
        N_tbins (int): Number of time bins (frame)
        N_edhbins (int): Number of EDH bins
        seed (int, optional): Choose the random seed. Defaults to 0.
        save_traj (bool, optional): Set the flag to save binner trajectories. Defaults to True.
        pix_r (int, optional): Choose row number of the pixel to save trajectory. Defaults to 0.
        pix_c (int, optional): Choose column number of the pixel to save trajectory. Defaults to 0.
        step_params (dict, optional): Dictionary to pass different stepping parameters. Defaults to {}.
    """ 
    BaseEDHSPC.__init__(self,Nr, Nc, N_pulses, device, N_tbins, N_edhbins, seed=seed)
    if not(step_params):
      # Set the default stepping params
      self.k = 1
    else:
      self.k = step_params["k"]
      
    self.save_traj = save_traj
    self.init_edh_params()
    self.temp_delta = 0
    self.pix_r = pix_r
    self.pix_c = pix_c

    
  def init_edh_params(self):
    r"""This method is called in the constructor and ensures initializing all the necessary common tensors that will be updated later.
    Initializing the tensors in the constructor eliminates the need to construct them at each laser cycle hence reduces simulation time.
    """
    self.e1 = torch.zeros((self.Nr, self.Nc, self.N_edhbins-1)).to(torch.float).to(self.device)
    self.pl = torch.zeros_like(self.e1).to(self.device)
    self.pe = torch.zeros_like(self.e1).to(self.device)
    self.prev_step = torch.zeros_like(self.e1).to(self.device)
    self.prev_err = torch.zeros_like(self.e1).to(self.device)
    self.delu = torch.zeros_like(self.e1).to(self.device)
    self.e = self.e1.to(torch.long).to(self.device)
    self.cy_cnt = 0
    self.kp = 0
    self.delta = 0
    self.eps = self.N_tbins*10e-8

    self.ch = torch.arange(self.N_edhbins-1).reshape(1, 1, self.N_edhbins-1)+1
    self.ch = torch.tile(self.ch, (self.Nr, self.Nc,1))
    self.a = (self.ch*1.0/(self.N_edhbins)).reshape(self.Nr, self.Nc, self.N_edhbins-1).to(self.device)
    self.b = ((self.N_edhbins - self.ch)*1.0/(self.N_edhbins)).reshape(self.Nr, self.Nc, self.N_edhbins-1).to(self.device)

    # Note: index values are starting from 1 till n_tbins
    self.time_idx = (torch.arange(self.N_tbins)+1).to(device = self.device,
                                                      dtype = torch.long)

    self.rr,self.cc  = torch.meshgrid(torch.linspace(0,self.Nr-1,self.Nr),
                                          torch.linspace(0,self.Nc-1,self.Nc), indexing='ij')
    self.rr = self.rr.to(device = self.device,
                                        dtype = torch.long)
    self.cc = self.cc.to(device = self.device,
                                        dtype = torch.long)
    self.rr2 = self.rr.reshape(self.Nr, self.Nc,1)
    self.cc2 = self.cc.reshape(self.Nr, self.Nc,1)
    self.idx = torch.arange(self.N_edhbins-1)
    self.i = self.idx.repeat(self.Nr,self.Nc,1)
    self.reset_edh()
    self.set_decay_schedule() 

  def reset_edh(self):
    r"""This method is called once in the constructor to decide the initialization scheme of the binners at the first laser cycle. 
    """
    temp = (((torch.arange(self.N_edhbins-1, dtype=torch.float)+1.0)*1.0/self.N_edhbins)*self.N_tbins).to(torch.long)
    self.e[self.rr2,self.cc2,self.i] = temp.to(self.device)
    self.e1 = self.e.to(torch.float)

  def update_pa_pb_kp(self, hist, ts):
    r"""Method to compute the early and late photon streams for current control value of each binner in a vectorized form.

    Refer HEDHBaseClass for more details about early and late photons

    Args:
        hist (torch.tensor): Input one-hot encoded photon detection cube.
        ts (torch.tensor): Time stamp tensor of the same size as hist. ts[n,m,k] = k if hist[n,m,k] = 1 else =0
    """
    # time stamps
    self.kp = self.kp*0
    self.pl = self.pl*0
    self.pe = self.pe*0

    self.kp = torch.sum(hist, axis=2).reshape(self.Nr, self.Nc, 1).to(self.device)

    for i in range(self.N_edhbins-1):
      pa_mask = ts > self.e1[:,:,i].clone().reshape(self.Nr, self.Nc, 1).to(self.device)
      pb_mask = (ts < self.e1[:,:,i].clone().reshape(self.Nr, self.Nc, 1).to(self.device))*(ts>0)
      # NOTE: This implementation can also handle cases when more than one photons are detected
      self.pl[:,:,i] = torch.sum(pa_mask*hist, axis=2)
      self.pe[:,:,i] = torch.sum(pb_mask*hist, axis=2)

  def update_delta(self):
    r"""Method to update the difference between early photons (``pe``) and late photons (``pl``). 
    The `delta` value is used to compute the binner update step. For proportional binners early and 
    late photons are multiplied by respective quantile fractions for proportional stepping explained in 
    [2].
    
    For a basic proportional binner delta is used to compute the binner update step as ``step[n+1] = delta[n]``. 
    """

    self.delta = (self.a*self.pl - self.b*self.pe).to(self.device)
    self.delta = torch.divide(self.delta, self.kp+self.eps)

  def apply_edh_step(self):
    r"""Method to apply the computed step size to update the control values of respective binners.
    """

    self.e1 = self.e1 + self.prev_step
    self.e1 = torch.clip(self.e1, 0, self.N_tbins)
    self.e = self.e1.to(device=self.device, dtype = torch.long)
  
  def set_decay_schedule(self):
    r""" Method to set the stepping schedule for the binners. 
    
    The ```decay_schedule``` contains a list of decay values for each laser cycle. Hence its length is equal to the number of laser cycles. 
    A simple example of could be to overwrite this method with the following definition

    .. code:: python

        def set_decay_schedule(self):
          self.decay_schedule = np.ones(self.N_pulses).tolist()
    """
    self.decay_schedule = np.ones(self.N_pulses).tolist()

  def update_edh(self, hist):
    r"""Method to compute and apply the update step to the binners

    This method obtains the decay value for the specific cycle (cy_cnt), updates the delta_mask to choose valid binners to update at (cy_cnt),
    generates timestamp vectors from photon detection vectors, uses the timestamp vectors to compute before and after photons for each binner, 
    compute the delta (late photons - early photons)/total photons, computer the update step for each binner and apply the update step to corresponding binners.
    """

    self.decay = self.decay_schedule[self.cy_cnt]

    ts, hist = self.get_ts_from_hist(hist)

    self.update_pa_pb_kp(hist, ts)

    self.update_delta()

    new_step = (self.delta*self.k).to(self.device)
    
    self.prev_step = new_step*self.decay
    
    self.apply_edh_step()
    
    self.cy_cnt+=1

    return self.e1 
  

  def capture(self, phi_bar):
    r"""Method to generate SPC data for average incident photon flux (phi_bar) for 
    given number of laser cycles and other SPC intrinsic parameters.

    Args:
        phi_bar (torch.tensor): Average incident photon flux tensor of shape (Nr, Nc, N_tbins)

    Returns:
        captured_data (dictionary): Dictionary containing ED histogram for detected photons, the oracle and ground truth ED histograms, corresponding EW histogram tensor and 
        list of binner control values tracked by binner at each laser cycle for a pixel at ``(self.pix_r, self.pix_c)``. 
    """
    
    # print(phi_bar[0,0,:])
    self.reset_edh()
    traj = []
    ewh_img = 0
    for i in tqdm(range(self.N_pulses)):
      hist = self.sim_poisson_process(phi_bar)
      ewh_img+=hist
      edh_img = self.update_edh(hist)
      if self.save_traj:
        traj.append(edh_img[self.pix_r,self.pix_c,:].cpu().tolist())
    

    edh_bins = self.e1
    edh_bins = self.add_extreme_boundaries(edh_bins)

    oedh_bins = self.ewh2edh(ewh_img)
    oedh_bins = self.add_extreme_boundaries(oedh_bins)

    gt_edh_bins = self.ewh2edh(phi_bar)
    gt_edh_bins = self.add_extreme_boundaries(gt_edh_bins)

    captured_data = {
        'edh':edh_bins,
        'oedh':oedh_bins,
        'gtedh':gt_edh_bins,
        'ewh':ewh_img,
        'traj':traj,
    }
    return captured_data


class PEDHOptimized(PEDHBaseClass):
  def __init__(self, 
               Nr, 
               Nc, 
               N_pulses, 
               device, 
               N_tbins, 
               N_edhbins, 
               seed=0, 
               save_traj = True, 
               pix_r = 0, 
               pix_c = 0, 
               step_params = {}):
    
    r"""
    Inherits the :class:`.PEDHBaseClass`. Overwritten methods are `set_decay_schedule` and `update_edh`.

    Addtionally the following default stepping parameters are used

    .. code:: python

              step_params = {
                  "k":3, # Step size gain
                  "decay":0,
                  "mtm":0.8,
                  "min_clip":0.02,
                  "switch_fraction":0.8,
                  "delta_mem": 0.95
              }
    
    .. note:: The value of delta is set to zero because it is computed later using the following formula
              `min_clip**(1/(N_pulses*switch_fraction)`.

    """

    if not(step_params):
      step_params = {
                  "k":3, # Step size gain
                  "decay":0,
                  "mtm":0.8,
                  "min_clip":0.02,
                  "switch_fraction":0.8,
                  "delta_mem": 0.95
              }
    
    self.k = step_params["k"]
    self.decay = step_params["decay"]
    self.mtm = step_params["mtm"]
    self.min_clip = step_params["min_clip"]
    self.switch_fraction = step_params["switch_fraction"]
    self.delta_mem = step_params["delta_mem"]

    PEDHBaseClass.__init__(
                            self,
                            Nr, 
                            Nc,  
                            N_pulses, 
                            device, 
                            N_tbins,
                            N_edhbins,
                            seed = seed, 
                            save_traj = True, 
                            pix_r = pix_r, 
                            pix_c = pix_c,
                            step_params=step_params)
    
    
  def set_decay_schedule(self):
    r""" Method to generate improved decay schedule based on the optimized stepping strategy [2].
    """
    if not (self.decay):
        self.decay = self.min_clip**(1/(self.N_pulses*self.switch_fraction))

    decay_schedule = []
    for cy_cnt in range(self.N_pulses):
      if cy_cnt < self.switch_fraction*self.N_pulses:
        d1 = (self.decay)**cy_cnt
      else:
        d1 = self.min_clip
      decay_schedule.append(d1)
    
    self.decay_schedule = np.array(decay_schedule)
  
  def update_edh(self, hist):
    r""" Update method applying temporal decay and temporal smoothing and scaling based on 
    optimized stepping strategy for PEDH.
    """

    ts, hist = self.get_ts_from_hist(hist)

    self.update_pa_pb_kp(hist, ts)

    self.update_delta()

    # Applying temporal smoothing on delta
    self.temp_delta = self.delta*(1-self.delta_mem) + self.temp_delta*self.delta_mem

    # Applying scaling on step size
    new_step = (self.temp_delta*self.N_tbins*self.k*1.0/100.0).to(self.device)

    # Applying temporal smoothing and decay on the step size
    self.prev_step = ((1-self.mtm)*new_step + self.mtm*self.prev_step)*(self.decay_schedule[self.cy_cnt])
    
    # Appying the final update step
    self.apply_edh_step()

    self.cy_cnt+=1

    return self.e1 


"""
**********
References
**********

[1] A. Ingle and D. Maier, "Count-Free Single-Photon 3D Imaging with Race Logic," in IEEE Transactions on Pattern Analysis and Machine Intelligence, doi: 10.1109/TPAMI.2023.3302822.
keywords: {Histograms;Photonics;Cameras;Three-dimensional displays;Sensors;Laser pulses;Bandwidth;Computational Photography;3D Sensing;SPAD LiDAR;Race Logic},

[2] Sadekar, K., Maier, D., & Ingle, A. (2025). Single-Photon 3D Imaging with Equi-Depth Photon Histograms. In European Conference on Computer Vision (pp. 381-398). Springer, Cham.
"""