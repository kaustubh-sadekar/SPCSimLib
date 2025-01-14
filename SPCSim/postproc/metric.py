import numpy as np

def rmse(pred, gt):
  err_img = np.sqrt((pred - gt)**2)
  err = np.sqrt(np.mean((pred - gt)**2)) # np.mean(err_img)
  return err_img, err

def median_ae(pred, gt):
  err_img = np.abs(pred - gt)
  err = np.median(err_img)
  return err_img, err


def ame(pred, gt):
  err_img = np.abs(pred - gt)
  err = np.mean(err_img)
  return err_img, err

def p_inlier(pred, gt, alpha):
  r"""
  alpha is in % of gt
  NOTE: It is in % so in the code we multiply 0.01 to the final value
  """
  temp_img = np.abs(pred - gt)
  temp_img = np.divide(temp_img, gt)
  in_mask = (temp_img < alpha*0.01).astype(np.uint8)
  err = np.sum(in_mask)/(in_mask.shape[0]*in_mask.shape[1])
  return in_mask, err*100

def p_inlier2(pred, gt, alpha):
  r"""
  alpha is in mm
  """
  temp_img = np.abs(pred - gt)
  in_mask = (temp_img < alpha).astype(np.uint8)
  err = np.sum(in_mask)/(in_mask.shape[0]*in_mask.shape[1])
  return (temp_img >= alpha).astype(np.uint8), err*100
