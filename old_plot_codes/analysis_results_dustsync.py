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
from analys_lib import plotr_gaussproduct, plotmed
from matplotlib.backends.backend_pdf import PdfPages 

r=0.
nside = 64
lmax = nside*3-1
#lmax=850
scale = 10
Nlbin = 10
fsky = 0.7
ELLBOUND = 15
dusttype = 0
synctype = 0
kw=''
kwsim=''
Pathload='./'

b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
l = b.get_effective_ells()
l = l[:ELLBOUND]
Nell = len(l)

res1=np.load('Best-fits/resultso1bt_PL_d%ss%sc_fix0.npy'%(0,0),allow_pickle=True).item()
res2=np.load('Best-fits/resultso1bt_PL_d%ss%sc_fix0.npy'%(0,0),allow_pickle=True).item()
#res2=np.load('Best-fits/resultso1bt_PL_d%ss%sc_fix1.npy'%(dusttype,synctype),allow_pickle=True).item()

legs1='d0s0'
legs2='d1s1'

c1='darkblue'
c2='darkorange'

fig1=plt.figure()
plotmed(l,'X2red',res1,show=False,color=c1,legend=legs1)
plotmed(l+2,'X2red',res2,show=False,color=c2,legend=legs2)
plt.plot(l,np.ones(len(l)),c='k',linestyle='--')
#plt.show()

fig2=plt.figure()
plotmed(l+1,'A_s',res1,show=False,color=c1,legend=legs1)
plotmed(l+2,'A_s',res2,show=False,color=c2,legend=legs2)
#plt.show()

fig3=plt.figure()
plotmed(l,'A_sd',res1,show=False,legend=legs1,color=c1)
plotmed(l+2,'A_sd',res2,show=False,color=c2,legend=legs2)
#plt.show()

fig4=plt.figure()
plotmed(l,'beta_s',res1,show=False,color=c1,legend=legs1)
plotmed(l+2,'beta_s',res2,show=False,color=c2,legend=legs2)
plt.plot(l,-3*np.ones(len(l)),c='k',linestyle='--')
#plt.show()

fig5=plt.figure()
plotmed(l+1,'beta',res1,show=False,color=c1,legend=legs1)
plotmed(l+2,'beta',res2,show=False,color=c2,legend=legs2)
#plotmed(l,'beta',res2,show=False,color=c2,legend=legs2)
#plt.show()

fig6=plt.figure()
plotmed(l+1,'temp',res1,show=False,color=c1,legend=legs1)
plotmed(l+2,'temp',res2,show=False,color=c2,legend=legs2)
#plt.show()

fig7=plt.figure()
plotmed(l+1,'r',res1,show=False,color=c1,legend=legs1)
plotmed(l+2,'r',res2,show=False,color=c2,legend=legs2)
plt.plot(l,-3*np.zeros((len(l))),c='k',linestyle='--')
#plt.show()

fig8=plt.figure()
plotmed(l+2,'Aw1b',res1,show=False,color=c1,legend=legs1)
plotmed(l,'Aw1b',res2,show=False,color=c2,legend=legs2)
#plt.show()

fig9=plt.figure()
plotmed(l+2,'Aw1t',res1,show=False,color=c1,legend=legs1)
plotmed(l,'Aw1t',res2,show=False,color=c2,legend=legs2)
#plt.show()

fig11=plt.figure()
plotmed(l+2,'w1bw1t',res1,show=False,color=c1,legend=legs1)
plotmed(l,'w1bw1t',res2,show=False,color=c2,legend=legs2)
#plt.show()

fig16=plt.figure()
plotmed(l+2,'w1bw1b',res1,show=False,color=c1,legend=legs1)
plotmed(l,'w1bw1b',res2,show=False,color=c2,legend=legs2)
#plt.show()

fig11=plt.figure()
plotmed(l+2,'w1tw1t',res1,show=False,color=c1,legend=legs1)
plotmed(l,'w1tw1t',res2,show=False,color=c2,legend=legs2)
#plt.show()

fig12=plt.figure()
plotmed(l+2,'Asw1b',res1,show=False,color=c1,legend=legs1)
plotmed(l,'Asw1b',res2,show=False,color=c2,legend=legs2)
#plt.show()

fig13=plt.figure()
plotmed(l+2,'Asw1t',res1,show=False,color=c1,legend=legs1)
plotmed(l,'Asw1t',res2,show=False,color=c2,legend=legs2)
#plt.show()

fig14=plt.figure()
plotr_gaussproduct(res1,Nmin=0,Nmax=15,debug=False,color=c1,save=True)
fig15=plt.figure()
plotr_gaussproduct(res2,Nmin=0,Nmax=15,debug=False,color=c2,save=True)

def save_image(filename): 
    
    # PdfPages is a wrapper around pdf  
    # file so there is no clash and create 
    # files with no error. 
    p = PdfPages(filename) 
      
    # get_fignums Return list of existing  
    # figure numbers 
    fig_nums = plt.get_fignums()   
    figs = [plt.figure(n) for n in fig_nums] 
      
    # iterating over the numbers in list 
    for fig in figs:  
        
        # and saving the files 
        fig.savefig(p, format='pdf')  
      
    # close the object 
    p.close()   
 
filename = "%svs%s.pdf"%(legs1,legs2)  
  
# call the function 
save_image(filename)   