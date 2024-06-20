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
import matplotlib.patheffects as path_effects
import scipy.stats as st
import basicfunc as func
from fgbuster import get_instrument, get_sky, get_observation  # Predefined instrumental and sky-creation configurations
import analys_lib as an

dusttype=0
resultsmbb=np.load('Best-fits/resultsmbb_d%sc.npy'%dusttype,allow_pickle=True).item()
color='darkorange'

Nmin=0
Nmax=15
rl = resultsmbb['r']

sig=np.std(rl,axis=1)
sigan= np.sqrt(1/np.sum(1/(sig**2)))

