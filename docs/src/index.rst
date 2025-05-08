Getting started
===============

SPCSim (Single-Photon Camera Simulator) is a PyTorch-based library for simulating single-photon cameras (SPCs). 
The current version of SPCSim includes implementations of 3D SPCs used in direct time of flight (dToF) applications. 
SPCSim enables users to design customized 3D imaging pipelines and simulate SPC measurements for different scene distances and illumination conditions.


Key features of SPCSim
======================

* Vectorized PyTorch implementations with GPU acceleration
* Quick and easy to use API to encourage the research related to single photon cameras
* Reduced simulation and overall experimentation time
* Modular and customizable framework to simulate/emulate real senor data and SPC data processing pipelines
* PerPixelLoader class to simulate multiple independent runs, in parallel, for each combination of scene distances and illumination conditions.
* Implementation of different SPC data compression methods
* Implementation of postprocessing algorithms to estimate scene distances from compressed SPC measurements
* Implementation of customizable active illumination laser (approximated as a gaussian) with control over the laser FWHM, power, and repetition rate.
* Utility functions to plot and visualize different data modalities.


Citing SPCSim
=============

If you find this tool helpful, please cite our `paper <https://link.springer.com/chapter/10.1007/978-3-031-73039-9_22>`_:

.. code-block:: shell         
      
      @InProceedings{10.1007/978-3-031-73039-9_22,
      author="Sadekar, Kaustubh
      and Maier, David
      and Ingle, Atul",
      editor="Leonardis, Ale{\v{s}}
      and Ricci, Elisa
      and Roth, Stefan
      and Russakovsky, Olga
      and Sattler, Torsten
      and Varol, G{\"u}l",
      title="Single-Photon 3D Imaging with Equi-Depth Photon Histograms",
      booktitle="Computer Vision -- ECCV 2024",
      year="2025",
      publisher="Springer Nature Switzerland",
      address="Cham",
      pages="381--398",
      isbn="978-3-031-73039-9"
      }


Installation guide
==================

1. Create a new virtual environment and install the required libraries

.. code-block:: shell
   
      git clone https://github.com/kaustubh-sadekar/SPCSimLib.git
      cd SPCSimLib/
      conda create -n "spcsim_env" python=3.9.12
      conda activate spcsim_env
      pip install -r requirements.txt

2. Install SPCSim

.. code-block:: shell
      
         pip install .


Acknowledgements
================

This work was supported in part by NSF ECCS-2138471.

.. image:: https://upload.wikimedia.org/wikipedia/commons/1/11/NSF_Official_logo_Med_Res.png
   :height: 200
   :width: 200


.. note:: To simulate more advanced active SPC measurements please refer to |visionsim_link_activespc|. |visionsim_link_index| is a modular and extensible framework that realistically emulates many different sensor types, alongside rich pixel-perfect ground truth annotations across low-, mid-, and high-level scene characteristics, as well as intrinsic and extrinsic camera properties. 


.. |visionsim_link_activespc| raw:: html

   <a href="https://visionsim.readthedocs.io/en/latest/sections/sensors/spcs/active_spc.html" target="_blank">VisionSIM documentation </a>


.. |visionsim_link_index| raw:: html

   <a href="https://visionsim.readthedocs.io/en/latest/index.html" target="_blank">VisionSIM</a>


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting started

   self

.. toctree::
   :maxdepth: 2
   :hidden:  
   :caption: Tutorials:

   ../notebooks/DataLoaders
   ../notebooks/DToFSensors

.. toctree::
   :maxdepth: 2
   :hidden:  
   :caption: Advanced Tutorials:

   ../notebooks/Custom3DSensingPipeline
   ../notebooks/TutorialCreateCustomEDH

.. toctree::
   :maxdepth: 2
   :hidden:  
   :caption: Research Paper Implementations:

   ../notebooks/DeePEDH_ECCV24
   ../notebooks/Simulate_SwisSPAD2


.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: API Documentation

   modules
