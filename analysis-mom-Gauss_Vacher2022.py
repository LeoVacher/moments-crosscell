import sys
sys.path.append("./lib")

import pymaster as nmt 
import pysm3
import time
from mpfit import mpfit
import mpfitlib as mpl
import scipy
#from Nearest_Positive_Definite import *
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import matplotlib.patheffects as path_effects
import scipy.stats as st
import basicfunc as func
from fgbuster import get_instrument, get_sky, get_observation  # Predefined instrumental and sky-creation configurations
import analys_lib as an


#load dictionnaries:

resultsmbb =np.load('Best-fits/resultsmbb_d%sc.npy'%dusttype,allow_pickle=True).item()
resultso1b =np.load('Best-fits/resultso1b_d%sc.npy'%dusttype,allow_pickle=True).item()
resultso1bt =np.load('Best-fits/resultso1bt_d%sc.npy'%dusttype,allow_pickle=True).item()
resultso2b =np.load('Best-fits/resultso2b_d%sc.npy'%dusttype,allow_pickle=True).item()


