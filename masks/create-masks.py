import sys
sys.path.append("./lib")

import numpy as np
import healpy as hp
import pymaster as nmt 
import pysm3
import time
from mpfit import mpfit
import mpfitlib as mpl
import scipy
#from Nearest_Positive_Definite import *
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as path_effects
import scipy.stats as st
import basicfunc as func
import analys_lib as an
import simu_lib as sim
import pysm3.units as u

fsky=0.8
scale=10
nside=64

fskylist=np.array([0.2,0.4,0.6,0.7,0.8,0.9,0.97,0.99])
field= list(np.where(fskylist==fsky)[0])
mask0 = hp.read_map("./masks/HFI_Mask_GalPlane-apo0_2048_R2.00.fits", field=field)
mask0 = nmt.mask_apodization(mask0, scale, apotype="C2")
mask = hp.pixelfunc.ud_grade(mask0, nside, dtype=float)

#hp.mollview(mask)

hp.write_map("./masks/mask_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale), mask, dtype=float, overwrite=True)