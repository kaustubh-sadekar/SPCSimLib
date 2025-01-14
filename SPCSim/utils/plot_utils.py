import sys
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(".")
import numpy as np
import torch


def binlvl(idx,edh_len):
    r"""Funciton to calculate and return the binning level for each index.

    Args:
        idx (int): EDH bin index
        edh_len (int): Total ED histogram bins

    Returns:
        bdry_level: Boundary level
    """
    
    bdry_level = int(math.log(edh_len+1,2) - math.log(math.gcd(idx+1, edh_len+1),2)) - 1
    
    return bdry_level


def plot_edh(ebins_full, ax_, tr = None, plt_type = '-g', crop_window = 0, ymax = 0.5, lw = 1, ls='-', colors_list=['r','g','b','y','c','m']):
    r"""Function to plot the ED histogram boundaries based on the levels for better understanding.

    .. note:: This function supports plotting of bins only till 6th level i.e. till 64 bins

    Args:
        ebins_full (np.ndarray): ED histogram bins with the boundary bins
        ax_ (_type_): Matplotlib plotting axis
        tr (np.ndarray, optional): True transient or EW histogram with bins = ebins_full[-1]. Defaults to None.
        plt_type (str, optional): Used to change the line plot color and style for tr. Defaults to '-g'.
        crop_window (int, optional): Allows to zoom-in closer to the peak. Defaults to 0.
        ymax (float, optional): Height of ED histogram bins. Defaults to 0.5.
        lw (int, optional): Change the line width of ED histogram boundaries. Defaults to 1.
        ls (str, optional): Change the line style of ED histogram boundaries. Defaults to '-'.
        colors_list (list, optional): List of color values used for ED boundaries of different levels. Defaults to ['r','g','b','y','c','m'].
    """
    
    
    if not len(colors_list):
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
    else:
        colors = colors_list
    
    ebins = ebins_full[1:-1]

    for idx,e in enumerate(ebins):
        c = colors[binlvl(idx,len(ebins))]
        ax_.vlines(x = e,
                    ymax=ymax,
                    ymin=0,
                    color = c,
                    linewidth = lw,
                    linestyle=ls)
    
    if ls=='-':
    
        ax_.vlines(x = ebins_full[0],
                    ymax=ymax,
                    ymin=0,
                    color = 'k',
                    linewidth = lw,
                    linestyle=ls)
        
        ax_.vlines(x = ebins_full[-1],
                    ymax=ymax,
                    ymin=0,
                    color = 'k',
                    linewidth = lw,
                    linestyle=ls)
        
        ax_.hlines(y = ymax,
                    xmax=ebins_full[-1],
                    xmin=0,
                    color = 'k',
                    linewidth = lw,
                    linestyle=ls)
        
        ax_.hlines(y = 0,
                    xmax=ebins_full[-1],
                    xmin=0,
                    color = 'k',
                    linewidth = lw,
                    linestyle=ls)
    
    if tr is not None:
        assert isinstance(tr, np.ndarray), "Ensure transient is a numpy array before passing to plot_transient function"
        assert (len(tr.shape) == 1), "Ensure transient is a 1D array"

        plot_transient(ax_, tr, plt_type=plt_type)

        if crop_window:
            ax_.set_xlim(np.argmax(tr)-crop_window, np.argmax(tr)+crop_window)
    
    ax_.set_xlabel("Time (a.u.)")
    ax_.set_ylabel("Photon counts per bin")


def plot_transient(ax, tr, plt_type = '-g', label="True transient"):
    r"""Function to plot the true transients, photon densities and EW histograms with N_ewhbins = N_tbins 

    Args:
        ax_ (_type_): Matplotlib plotting axis
        tr (np.ndarray, optional): True transient or EW histogram with bins = ebins_full[-1]. Defaults to None.
        plt_type (str, optional): Used to change the line plot color and style for tr. Defaults to '-g'.
        label (str, optional): Plot label for tr. Defaults to "True transient".
    """

    assert isinstance(tr, np.ndarray), "Ensure transient is a numpy array before passing to plot_transient function"
    assert (len(tr.shape) == 1), "Ensure transient is a 1D array"

    ax.plot(tr, plt_type, label = label)


def plot_rgbd(fig, ax_rgb, ax_dist, rgb, dist, cmap="gray", axis="off", rgb_title = "RGB image", show_cbar = False):
    r"""Function to plot RGB and Distance images

    Args:
        fig (_type_): Matplotlib figure object
        ax_rgb (_type_): Matplotlib subplot axis for RGB image
        ax_dist (_type_): Matplotlib subplot axis for distance image
        rgb (_type_): RGB image
        dist (_type_): Distance image
        cmap (str, optional): Color map used to display the distance image. Defaults to "gray".
        axis_off (str, optional): Remove image axis. Defaults to "off".
        rgb_title (str, optional): Title for rgb image. Defaults to "RGB image".
        show_cbar (bool, optional): Flag to select displaying color bar. Defaults to False.

    Returns:
        im: Matplotlib imshow output object for further use.
    """
  
    assert isinstance(rgb, np.ndarray), "Ensure rgb is a numpy array before passing to plot_transient function"
    assert isinstance(dist, np.ndarray), "Ensure dist is a numpy array before passing to plot_transient function"
    assert (len(dist.shape) == 2), "Ensure transient is a 1D array"

    ax_rgb.imshow(rgb)
    ax_rgb.axis(axis)
    ax_rgb.set_title(rgb_title)

    im = ax_dist.imshow(dist, cmap = cmap)
    ax_dist.axis(axis)
    ax_dist.set_title("Distance map")
    
    # displaying colorbar
    if show_cbar:
      divider = make_axes_locatable(ax_dist)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      cbar = fig.colorbar(im, cax=cax, orientation='vertical')

      cbar.ax.tick_params(length=0)
      cbar.set_label('Distance (m)', rotation=270, labelpad=15)

    return im

def plot_ewh(ax, xbins, ewh, label=None, color = None):
    r"""Function to plot EW histograms

    Args:
        ax_ (_type_): Matplotlib plotting axis
        xbins (_type_): EW histogram midpoint locations on time axis.
        ewh (_type_): EW histogram
        label (_type_, optional): Plot label for EW histogram. Defaults to None.
        color (_type_, optional): EW histogram edge color. Defaults to None.
    """

    if color is None:
      bar_c = 'b'
    else:
      bar_c = color

    assert isinstance(ewh, np.ndarray), "Ensure ewh is a numpy array before passing to plot_transient function"
    assert (len(ewh.shape) == 1), "Ensure ewh is a 1D array"

    if label is not None:
      ax.bar(xbins, ewh, label = label, width=(xbins[1]-xbins[0]), align="edge", color = bar_c, edgecolor="k")
    else:
      ax.bar(xbins, ewh, width=(xbins[1]-xbins[0]), align="edge", color = bar_c, edgecolor="k")


    
def plot_edh_traj(ax, edh_list, gt_edh_list, tr, ewh = None, colors = None):
    r"""Function to plot the EDH binner trajectories

    Args:
        ax_ (_type_): Matplotlib plotting axis
        edh_list (_type_): List of EDH binner CV values for each laser cycle
        gt_edh_list (_type_): List of true quantiles to be tracked by the EDH binners
        tr (np.ndarray, optional): True transient or EW histogram with bins = ebins_full[-1]. Defaults to None.
        colors (list, optional): List of color values used for ED boundaries of different levels. Defaults to None.
    """

    assert edh_list.shape[-1] == gt_edh_list.shape[-1]

    print(edh_list.shape)
    print(gt_edh_list.shape)

    if colors == None:
        colors=['r','g','b','y','c','m']

    for i in range(gt_edh_list.shape[-1]):
        c = colors[binlvl(i,(gt_edh_list.shape[-1]))]

        ax.hlines(y = gt_edh_list[i],
                xmax=edh_list.shape[0],
                xmin=0,
                color = c,
                linestyle='--')
    
        ax.plot(edh_list[:,i],'-'+c)
    
    tr_bins = np.linspace(0, tr.shape[-1], tr.shape[-1])
    ax.plot(tr,tr_bins, '-g', label="True Transient")
    
    if ewh is not None:
        ax.plot(ewh,tr_bins, '-b', label="EWH")
    
    ax.set_xlabel("Laser cycles")
    ax.set_ylabel("CV trajectories")