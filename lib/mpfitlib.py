from astropy import constants as constants
#import scipy.constants as constants
import numpy as np
import basicfunc as func
import pysm3
import healpy as hp
import pymaster as nmt
import numpy as np
import pysm_common as psm 
import basicfunc as func

#contains all the functions to be fitted by mpfit 

nside = 64
lmax = nside*3-1
Nlbin = 10 
ELLBOUND = 19

b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
l = b.get_effective_ells()

CLcmb_or=hp.read_cl('./CLsimus/Cls_Planck2018_r0.fits') #TT EE BB TE
CL_tens=hp.read_cl('./CLsimus/Cls_Planck2018_tensor_r1.fits')

DL_lensbin = l*(l+1)*b.bin_cell(CLcmb_or[2,2:lmax+3])[0:ELLBOUND]/2/np.pi
DL_tens = l*(l+1)*b.bin_cell(CL_tens[2,2:lmax+3])[0:ELLBOUND]/2/np.pi
 
def Gaussian(p,fjac=None, x=None, y=None, err=None):
    # Gaussian curve
    model = func.Gaussian(x,p[0],p[1])
    status = 0
    return([status, (y-model)/err])

def func_d_o0(p,fjac=None, x1=None, x2=None, y=None, err=None,nuref=353.):
    #fit function dust, order 0

    nu_i=x1
    nu_j=x2
    mbb = p[0] * func.mbb_uK(nu_i, p[1], p[2]) * func.mbb_uK(nu_j, p[1], p[2])
    model = mbb  + DL_lensbin[int(p[4])] + p[3] * DL_tens[int(p[4])]
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

def func_ds_o0(p,fjac=None, x1=None, x2=None, y=None, err=None,nuref=353.,nurefs=23.,ell=None):
    #fit function dust+syncrotron, order 0
    nu_i=x1
    nu_j=x2
    mbb = p[0] * func.mbb_uK(nu_i, p[1], p[2],nu0=nuref) * func.mbb_uK(nu_j, p[1], p[2],nu0=nuref)
    sync = p[3] * func.PL_uK(nu_i, p[4],nu0=nurefs) * func.PL_uK(nu_j, p[4],nu0=nurefs)
    crossdustsync = p[5] * np.sqrt(p[0]*p[3]) * (func.mbb_uK(nu_i, p[1], p[2],nu0=nuref) * func.PL_uK(nu_j, p[4],nu0=nurefs) + func.PL_uK(nu_i, p[4],nu0=nurefs) * func.mbb_uK(nu_j, p[1], p[2],nu0=nuref))
    model = mbb + sync + crossdustsync + DL_lensbin[ell] + p[6] * DL_tens[ell]
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

def func_ds_o1bt(p,fjac=None, x1=None, x2=None, y=None, err=None,nuref=353,nurefs=23.,ell=None):
    #fit function dust+syncrotron, order 1 in beta and T
    nu_i=x1
    nu_j=x2

    ampl = func.mbb_uK(nu_i,p[1],p[2],nu0=nuref)*func.mbb_uK(nu_j,p[1],p[2],nu0=nuref)
    sync= p[3]*func.PL_uK(nu_i,p[4],nu0=nurefs)*func.PL_uK(nu_j,p[4],nu0=nurefs)
    crossdustsync= p[5]*np.sqrt(p[0]*p[3])*(func.mbb_uK(nu_i,p[1],p[2],nu0=nuref)*func.PL_uK(nu_j,p[4],nu0=nurefs)+ func.PL_uK(nu_i,p[4],nu0=nurefs)*func.mbb_uK(nu_j,p[1],p[2],nu0=nuref))
    lognui = np.log(nu_i/nuref)
    lognuj = np.log(nu_j/nuref)
    dx0 = func.dmbbT(nuref,p[2])
    dxi = func.dmbbT(nu_i,p[2])
    dxj = func.dmbbT(nu_j,p[2])
    temp = ampl * (p[0]+ (lognui+lognuj) * p[6]+ lognui*lognuj * p[7])
    temp2= ampl * ((dxi+dxj-2*dx0)*p[8]+(lognuj*(dxi-dx0)+lognui*(dxj-dx0))*p[9]+(dxi-dx0)*(dxj-dx0)*p[10])
    crossdustsync2 = p[11]*(func.mbb_uK(nu_i,p[1],p[2],nu0=nuref)*lognui*func.PL_uK(nu_j,p[4],nu0=nurefs)+ func.PL_uK(nu_i,p[4],nu0=nurefs)*func.mbb_uK(nu_j,p[1],p[2],nu0=nuref)*lognuj)
    crossdustsync3 = p[12]*(func.mbb_uK(nu_i,p[1],p[2],nu0=nuref)*(dxi-dx0)*func.PL_uK(nu_j,p[4],nu0=nurefs)+ func.PL_uK(nu_i,p[4],nu0=nurefs)*func.mbb_uK(nu_j,p[1],p[2],nu0=nuref)*(dxj-dx0))
    model = temp + temp2 + sync+ crossdustsync+ crossdustsync2+ crossdustsync3+ DL_lensbin[ell] + p[13]*DL_tens[ell]
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

#to vectorize:

def func_ds_o1bts(p,fjac=None, x=None, y=None, err=None,nuref=353,nurefs=23.,ell=None):

    nu_i=x1
    nu_j=x2

    ampl = func.mbb_uK(nu_i,p[1],p[2])*func.mbb_uK(nu_j,p[1],p[2])
    sync= func.PL_uK(nu_i,p[4])*func.PL_uK(nu_j,p[4])
    crossdustsync= p[5]*np.sqrt(p[0]*p[3])*(func.mbb_uK(nu_i,p[1],p[2])*func.PL_uK(nu_j,p[4])+ func.PL_uK(nu_i,p[4])*func.mbb_uK(nu_j,p[1],p[2]))/(func.PL_uK(nurefs,p[4])*func.mbb_uK(nuref,p[1],p[2]))
    lognui = np.log(nu_i/nuref)
    lognuj = np.log(nu_j/nuref)
    lognuis = np.log(nu_i/nurefs)
    lognujs = np.log(nu_j/nurefs)
    dx0 = func.dmbbT(nuref,p[2])
    dxi = func.dmbbT(nu_i,p[2])
    dxj = func.dmbbT(nu_j,p[2])
    temp = ampl * (p[0]+ (lognui+lognuj) * p[6]+ lognui*lognuj * p[7])
    temp2 = ampl*((dxi+dxj-2*dx0)*p[8]+(lognuj*(dxi-dx0)+lognui*(dxj-dx0))*p[9]+(dxi-dx0)*(dxj-dx0)*p[10])
    syncmom = sync * (p[3]+ (lognuis+lognujs) * p[11]+ lognuis*lognujs * p[12])
    crossdustsync2 = p[13]*(func.mbb_uK(nu_i,p[1],p[2])*lognui*func.PL_uK(nu_j,p[4])+ func.PL_uK(nu_i,p[4])*func.mbb_uK(nu_j,p[1],p[2])*lognuj)/(func.PL_uK(nurefs,p[4])*func.mbb_uK(nuref,p[1],p[2]))
    crossdustsync3 = p[14]*(func.mbb_uK(nu_i,p[1],p[2])*(dxi-dx0)*func.PL_uK(nu_j,p[4])+ func.PL_uK(nu_i,p[4])*func.mbb_uK(nu_j,p[1],p[2])*(dxj-dx0))/(func.PL_uK(nurefs,p[4])*func.mbb_uK(nuref,p[1],p[2]))
    crossdustsync4 = p[15]*(func.mbb_uK(nu_i,p[1],p[2])*lognujs*func.PL_uK(nu_j,p[4])+ func.PL_uK(nu_i,p[4])*func.mbb_uK(nu_j,p[1],p[2])*lognuis)/(func.PL_uK(nurefs,p[4])*func.mbb_uK(nuref,p[1],p[2]))
    crossdustsync5 = p[16]*(func.mbb_uK(nu_i,p[1],p[2])*lognui*func.PL_uK(nu_j,p[4])*lognujs+ func.PL_uK(nu_i,p[4])*lognuis*func.mbb_uK(nu_j,p[1],p[2])*lognuj)/(func.PL_uK(nurefs,p[4])*func.mbb_uK(nuref,p[1],p[2]))
    crossdustsync6 = p[17]*(func.mbb_uK(nu_i,p[1],p[2])*(dxi-dx0)*func.PL_uK(nu_j,p[4])*lognujs+ func.PL_uK(nu_i,p[4])*lognuis*func.mbb_uK(nu_j,p[1],p[2])*(dxj-dx0))/(func.PL_uK(nurefs,p[4])*func.mbb_uK(nuref,p[1],p[2]))
    model[icross] = temp + temp2 + syncmom+ crossdustsync+ crossdustsync2+ crossdustsync3+crossdustsync4+crossdustsync5+crossdustsync6+ DL_lensbin[ell] + p[18]*DL_tens[ell]
    icross = icross + 1
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])
