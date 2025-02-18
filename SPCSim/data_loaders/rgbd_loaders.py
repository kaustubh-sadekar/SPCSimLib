import numpy as np
import cv2
import torch

class RGBDLoader:
  r"""General class to load RGB-D datasets
  """
  def __init__(self,
               Nr,
               Nc,
               crop_r1=0,
               crop_r2=-1,
               crop_c1=0,
               crop_c2=-1,
               device = 'cpu',
               n_tbins = 1000,
               tmax = 100, #time period in nano-seconds
               folder = 'train'
               ):
    """

    Args:
        Nr (int): Number of pixel rows in resized frame
        Nc (int): Number of pixel columns in resized frame
        crop_r1 (int, optional): Starting row to crop image. Defaults to 0.
        crop_r2 (int, optional): Ending row to crop image. Defaults to -1.
        crop_c1 (int, optional): Starting column to crop image. Defaults to 0.
        crop_c2 (int, optional): Ending column to crop image. Defaults to -1.
        device (str, optional): Choice of compute device. Defaults to 'cpu'.
        n_tbins (int, optional): Number of discrete time bins dividing the laser time period. Defaults to 1000.
        tmax (int, optional): Laser time period in nano seconds. Defaults to 100.

    .. note:: tmax is in nano seconds and not in seconds

    """


    self.dmax = 3*1e8*tmax*1e-9/2 # scene distance in meters dmax = (c*tmax)/2
    self.device = device
    self.folder = folder
    self.Nr = Nr
    self.Nc = Nc
    self.N_tbins = n_tbins
    self.crop_r1 = crop_r1
    self.crop_r2 = crop_r2
    self.crop_c1 = crop_c1
    self.crop_c2 = crop_c2
    self.loader_id = "RGBDLoader"
  
  def load_dist(self,dist_pth):
    r"""Method to load the distance image
    """

    dist_img = self.dist_preproc(cv2.resize(cv2.imread(dist_pth,-1)[self.crop_r1:self.crop_r2, self.crop_c1:self.crop_c2],(self.Nc, self.Nr)))
    return torch.tensor(dist_img).to(self.device)
  
  def load_rgb(self, rgb_pth):
    r"""Method to load the rgb image 
    
    .. note:: The color channels are flipped as cv2.imread reads bgr image instead of rgb
    
    """
    rgb_img = self.rgb_preproc(cv2.resize(cv2.imread(rgb_pth,1)[self.crop_r1:self.crop_r2, self.crop_c1:self.crop_c2,::-1],(self.Nc, self.Nr)))
    return torch.tensor(rgb_img).to(self.device)

  def load_albedo(self, albedo_pth, read_mode=0):
    r"""Method to load the albedo image 
    """
    albedo = self.albedo_preproc(cv2.resize(cv2.imread(albedo_pth,read_mode)[self.crop_r1:self.crop_r2, self.crop_c1:self.crop_c2],(self.Nc, self.Nr)))
    return torch.tensor(albedo).to(self.device)
  
  def load_intensity(self, intensity_pth, read_mode=0):
    r"""Method to load the intensity image 
    """
    intensity = self.intensity_preproc(cv2.resize(cv2.imread(intensity_pth,read_mode)[self.crop_r1:self.crop_r2, self.crop_c1:self.crop_c2],(self.Nc, self.Nr)))
    return torch.tensor(intensity).to(self.device)
  
  def rgb_preproc(self, rgb):
    r"""Method to preprocess the rgb image
    """
    return rgb
  
  def albedo_preproc(self, albedo):
    r"""Method to preprocess the albedo image
    """
    return albedo
  
  def intensity_preproc(self, intensity):
    r"""Method to preprocess the intensity image
    """
    return intensity
  
  def dist_preproc(self, dist):
    r"""Method to preprocess the distance image
    """
    return dist
  
  def get_data(self,rgb_pth,dist_pth,albedo_pth="",intensity_pth=""):
    r"""Method to get the RGB-D data

    Args:
        rgb_pth (str): File path to rgb image
        dist_pth (str): File path to distance image
        albedo_pth (str, optional): _description_. Defaults to "".
        intensity_pth (str, optional): _description_. Defaults to "".

    Returns:
        data (dictionary): Dictionary containing the rgb, intensity, albedo and distance image
    """
    
    rgb = self.load_rgb(rgb_pth)
    dist = self.load_dist(dist_pth)

    if albedo_pth == "":
      albedo = self.load_albedo(rgb_pth)
    else:
      albedo = self.load_albedo(albedo_pth)

    if intensity_pth == "":
      intensity = self.load_intensity(rgb_pth)
    else:
      intensity = self.load_intensity(intensity_pth)
    
    data = {
        'rgb':rgb.to(self.device),
        'albedo':albedo.to(self.device),
        'intensity':intensity.to(self.device),
        'gt_dist':dist.to(self.device),
        'loader_id':self.loader_id
    }

    return data

class NYULoader1(RGBDLoader):
  def __init__(self,
               Nr,
               Nc,
               crop_r1=44,
               crop_r2=470,
               crop_c1=40,
               crop_c2=600,
               device = 'cpu',
               n_tbins = 1000,
               tmax = 100, #time period in nano-seconds
               folder = 'train'
               ):
    r"""Data loader for NYUv2 dataset

    Args:
        Nr (int): Number of pixel rows in resized frame
        Nc (int): Number of pixel columns in resized frame
        crop_r1 (int, optional): Starting row to crop image. Defaults to 0.
        crop_r2 (int, optional): Ending row to crop image. Defaults to -1.
        crop_c1 (int, optional): Starting column to crop image. Defaults to 0.
        crop_c2 (int, optional): Ending column to crop image. Defaults to -1.
        device (str, optional): Choice of compute device. Defaults to 'cpu'.
        n_tbins (int, optional): Number of discrete time bins dividing the laser time period. Defaults to 1000.
        tmax (int, optional): Laser time period in nano seconds. Defaults to 100.

    .. note:: tmax is in nano seconds and not in seconds
    
    """
    
    RGBDLoader.__init__(self, Nr,
                        Nc,
                        crop_r1=crop_r1,
                        crop_r2=crop_r2,
                        crop_c1=crop_c1,
                        crop_c2=crop_c2,
                        device=device,
                        n_tbins=n_tbins,
                        tmax=tmax,
                        folder=folder)
    
    self.loader_id = "NYULoader1"

  def dist_preproc(self, dist):
    if self.folder == "test":
      dist = dist*10.0/(655.35*15.256)
    else:
      dist = dist*10.0/255.0
    
    return dist
