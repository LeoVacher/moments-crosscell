import scipy.constants as constants
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
    model = func.Gaussian(x,p[0],p[1])
    status = 0
    return([status, (y-model)/err])

def Fitdcordre0(p,fjac=None, x=None, y=None, err=None):
    ncross = len(x)
    nnus   = int((-1 + np.sqrt(ncross*8+1))/2.)
    posauto = [int(nnus*i - i*(i+1)/2 + i) for i in range(nnus)]
    nu = x[posauto]
    nuref=353.
    icross = 0
    model = np.zeros(ncross)
    for i in range(0,nnus):
        for j in range(i,nnus):
            model[icross] =  p[0]*func.mbb_uK(nu[i],p[1],p[2],nu0=nuref)*func.mbb_uK(nu[j],p[1],p[2],nu0=nuref) + DL_lensbin[int(p[4])]+ p[3]*DL_tens[int(p[4])]
            icross = icross + 1
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

def Fitdcordre0_vectorize(p,fjac=None, x1=None, x2=None, y=None, err=None,nuref=353.):
    nu_i=x1
    nu_j=x2
    mbb = p[0] * func.mbb_uK(nu_i, p[1], p[2]) * func.mbb_uK(nu_j, p[1], p[2])
    model = mbb  + DL_lensbin[int(p[4])] + p[3] * DL_tens[int(p[4])]
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

def Fitdscordre0(p,fjac=None, x=None, y=None, err=None):
    ncross = len(x)
    nnus   = int((-1 + np.sqrt(ncross*8+1))/2.)
    posauto = [int(nnus*i - i*(i+1)/2 + i) for i in range(nnus)]
    nu = x[posauto]
    nuref=353.
    nurefs=23.
    icross = 0
    model = np.zeros(ncross)
    for i in range(0,nnus):
        for j in range(i,nnus):
            mbb =  p[0]*(func.mbb_uK(nu[i],p[1],p[2])*func.mbb_uK(nu[j],p[1],p[2])/(func.mbb_uK(nuref,p[1],p[2])**2.)) 
            sync= p[3]*(func.PL_uK(nu[i],p[4])*func.PL_uK(nu[j],p[4])/(func.PL_uK(nurefs,p[4])**2))
            crossdustsync= p[5]*(func.mbb_uK(nu[i],p[1],p[2])*func.PL_uK(nu[j],p[4])+ func.PL_uK(nu[i],p[4])*func.mbb_uK(nu[j],p[1],p[2]))/(func.PL_uK(nurefs,p[4])*func.mbb_uK(nuref,p[1],p[2]))
            model[icross]=mbb+ sync + crossdustsync + DL_lensbin[int(p[7])]+ p[6]*DL_tens[int(p[7])]
            icross = icross + 1
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

def Fitdscordre0_vectorize(p,fjac=None, x1=None, x2=None, y=None, err=None,nuref=353.,nurefs=23.):
    nu_i=x1
    nu_j=x2
    mbb = p[0] * func.mbb_uK(nu_i, p[1], p[2],nu0=nuref) * func.mbb_uK(nu_j, p[1], p[2],nu0=nuref)
    sync = p[3] * func.PL_uK(nu_i, p[4],nu0=nurefs) * func.PL_uK(nu_j, p[4],nu0=nurefs)
    crossdustsync = p[5] * (func.mbb_uK(nu_i, p[1], p[2],nu0=nuref) * func.PL_uK(nu_j, p[4],nu0=nurefs) + func.PL_uK(nu_i, p[4],nu0=nurefs) * func.mbb_uK(nu_j, p[1], p[2],nu0=nuref))
    model = mbb + sync + crossdustsync + DL_lensbin[int(p[7])] + p[6] * DL_tens[int(p[7])]
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

def FitdcbetaT(p,fjac=None, x=None, y=None, err=None):
    nuref  = 353
    ncross = len(x)
    nnus   = int((-1 + np.sqrt(ncross*8+1))/2.)
    posauto = [int(nnus*i - i*(i+1)/2 + i) for i in range(nnus)]
    nu = x[0:ncross]
    nu = nu[posauto]
    icross = 0
    model = np.zeros(ncross)
    for i in range(nnus):
        for j in range(i,nnus):
                ampl = func.mbb_uK(nu[i],p[1],p[2])*func.mbb_uK(nu[j],p[1],p[2])
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognuj = np.log(nuj)
                dx0 = func.dmbbT(nuref,p[2])
                dxi = func.dmbbT(nu[i],p[2])
                dxj = func.dmbbT(nu[j],p[2])
                temp = ampl * (p[0]+ (lognui+lognuj) * p[3]+ lognui*lognuj * p[4])
                temp2=ampl*((dxi+dxj-2*dx0)*p[5]+(lognuj*(dxi-dx0)+lognui*(dxj-dx0))*p[6]+(dxi-dx0)*(dxj-dx0)*p[7])
                model[icross] = temp + temp2 + DL_lensbin[int(p[9])] + p[8]*DL_tens[int(p[9])]
                icross = icross + 1
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

def FitdcbetaT_vectorize(p,fjac=None, x1=None, x2=None, y=None, err=None,nuref=353):
    nu_i=x1
    nu_j=x2

    ampl = func.mbb_uK(nu_i,p[1],p[2],nu0=nuref)*func.mbb_uK(nu_j,p[1],p[2],nu0=nuref)
    sync=  p[3]*func.PL_uK(nu_i,p[4],nu0=nurefs)*func.PL_uK(nu_j,p[4],nu0=nurefs)
    lognui = np.log(nu_i/nuref)
    lognuj = np.log(nu_j/nuref)
    dx0 = func.dmbbT(nuref,p[2])
    dxi = func.dmbbT(nu_i,p[2])
    dxj = func.dmbbT(nu_j,p[2])
    temp = ampl * (p[0]+ (lognui+lognuj) * p[3]+ lognui*lognuj * p[4])
    temp2= ampl * ((dxi+dxj-2*dx0)*p[5]+(lognuj*(dxi-dx0)+lognui*(dxj-dx0))*p[6]+(dxi-dx0)*(dxj-dx0)*p[7])
    model = temp + temp2 + DL_lensbin[int(p[9])] + p[8]*DL_tens[int(p[0])]
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

def FitdscbetaT(p,fjac=None, x=None, y=None, err=None):
    nuref  = 353
    nurefs=23.
    ncross = len(x)
    nnus   = int((-1 + np.sqrt(ncross*8+1))/2.)
    posauto = [int(nnus*i - i*(i+1)/2 + i) for i in range(nnus)]
    nu = x[0:ncross]
    nu = nu[posauto]
    icross = 0
    model = np.zeros(ncross)
    for i in range(nnus):
        for j in range(i,nnus):
                ampl = func.mbb_uK(nu[i],p[1],p[2])*func.mbb_uK(nu[j],p[1],p[2])
                sync= p[3]*(func.PL_uK(nu[i],p[4])*func.PL_uK(nu[j],p[4])/(func.PL_uK(nurefs,p[4])**2))
                crossdustsync= p[5]*(func.mbb_uK(nu[i],p[1],p[2])*func.PL_uK(nu[j],p[4])+ func.PL_uK(nu[i],p[4])*func.mbb_uK(nu[j],p[1],p[2]))/(func.PL_uK(nurefs,p[4])*func.mbb_uK(nuref,p[1],p[2]))
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognuj = np.log(nuj)
                dx0 = func.dmbbT(nuref,p[2])
                dxi = func.dmbbT(nu[i],p[2])
                dxj = func.dmbbT(nu[j],p[2])
                temp = ampl * (p[0]+ (lognui+lognuj) * p[6]+ lognui*lognuj * p[7])
                temp2=ampl*((dxi+dxj-2*dx0)*p[8]+(lognuj*(dxi-dx0)+lognui*(dxj-dx0))*p[9]+(dxi-dx0)*(dxj-dx0)*p[10])
                crossdustsync2 = p[11]*(func.mbb_uK(nu[i],p[1],p[2])*lognui*func.PL_uK(nu[j],p[4])+ func.PL_uK(nu[i],p[4])*func.mbb_uK(nu[j],p[1],p[2])*lognuj)/(func.PL_uK(nurefs,p[4])*func.mbb_uK(nuref,p[1],p[2]))
                crossdustsync3 = p[12]*(func.mbb_uK(nu[i],p[1],p[2])*(dxi-dx0)*func.PL_uK(nu[j],p[4])+ func.PL_uK(nu[i],p[4])*func.mbb_uK(nu[j],p[1],p[2])*(dxj-dx0))/(func.PL_uK(nurefs,p[4])*func.mbb_uK(nuref,p[1],p[2]))
                model[icross] = temp + temp2 + sync+ crossdustsync+ crossdustsync2+ crossdustsync3+ DL_lensbin[int(p[14])] + p[13]*DL_tens[int(p[14])]
                icross = icross + 1
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

def FitdscbetaT_vectorize(p,fjac=None, x1=None, x2=None, y=None, err=None,nuref=353,nurefs=23.):
    nu_i=x1
    nu_j=x2

    ampl = func.mbb_uK(nu_i,p[1],p[2],nu0=nuref)*func.mbb_uK(nu_j,p[1],p[2],nu0=nuref)
    sync= p[3]*func.PL_uK(nu_i,p[4],nu0=nurefs)*func.PL_uK(nu_j,p[4],nu0=nurefs)
    crossdustsync= p[5]*(func.mbb_uK(nu_i,p[1],p[2],nu0=nuref)*func.PL_uK(nu_j,p[4],nu0=nurefs)+ func.PL_uK(nu_i,p[4],nu0=nurefs)*func.mbb_uK(nu_j,p[1],p[2],nu0=nuref))
    lognui = np.log(nu_i/nuref)
    lognuj = np.log(nu_j/nuref)
    dx0 = func.dmbbT(nuref,p[2])
    dxi = func.dmbbT(nu_i,p[2])
    dxj = func.dmbbT(nu_j,p[2])
    temp = ampl * (p[0]+ (lognui+lognuj) * p[6]+ lognui*lognuj * p[7])
    temp2= ampl * ((dxi+dxj-2*dx0)*p[8]+(lognuj*(dxi-dx0)+lognui*(dxj-dx0))*p[9]+(dxi-dx0)*(dxj-dx0)*p[10])
    crossdustsync2 = p[11]*(func.mbb_uK(nu_i,p[1],p[2],nu0=nuref)*lognui*func.PL_uK(nu_j,p[4],nu0=nurefs)+ func.PL_uK(nu_i,p[4],nu0=nurefs)*func.mbb_uK(nu_j,p[1],p[2],nu0=nuref)*lognuj)
    crossdustsync3 = p[12]*(func.mbb_uK(nu_i,p[1],p[2],nu0=nuref)*(dxi-dx0)*func.PL_uK(nu_j,p[4],nu0=nurefs)+ func.PL_uK(nu_i,p[4],nu0=nurefs)*func.mbb_uK(nu_j,p[1],p[2],nu0=nuref)*(dxj-dx0))
    model = temp + temp2 + sync+ crossdustsync+ crossdustsync2+ crossdustsync3+ DL_lensbin[int(p[14])] + p[13]*DL_tens[int(p[14])]
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

def FitdscbetaTbetas_full(p,fjac=None, x=None, y=None, err=None):
    nuref  = 353
    nurefs=23.
    ncross = len(x)
    nnus   = int((-1 + np.sqrt(ncross*8+1))/2.)
    posauto = [int(nnus*i - i*(i+1)/2 + i) for i in range(nnus)]
    nu = x[0:ncross]
    nu = nu[posauto]
    icross = 0
    model = np.zeros(ncross)
    for i in range(nnus):
        for j in range(i,nnus):
                ampl = func.mbb_uK(nu[i],p[1],p[2])*func.mbb_uK(nu[j],p[1],p[2])
                sync= func.PL_uK(nu[i],p[4])*func.PL_uK(nu[j],p[4])
                crossdustsync= p[5]*(func.mbb_uK(nu[i],p[1],p[2])*func.PL_uK(nu[j],p[4])+ func.PL_uK(nu[i],p[4])*func.mbb_uK(nu[j],p[1],p[2]))/(func.PL_uK(nurefs,p[4])*func.mbb_uK(nuref,p[1],p[2]))
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognuj = np.log(nuj)
                nuis = nu[i]/nurefs
                nujs = nu[j]/nurefs
                lognuis = np.log(nuis)
                lognujs = np.log(nujs)
                dx0 = func.dmbbT(nuref,p[2])
                dxi = func.dmbbT(nu[i],p[2])
                dxj = func.dmbbT(nu[j],p[2])
                temp = ampl * (p[0]+ (lognui+lognuj) * p[6]+ lognui*lognuj * p[7])
                temp2 = ampl*((dxi+dxj-2*dx0)*p[8]+(lognuj*(dxi-dx0)+lognui*(dxj-dx0))*p[9]+(dxi-dx0)*(dxj-dx0)*p[10])
                syncmom = sync * (p[3]+ (lognuis+lognujs) * p[11]+ lognuis*lognujs * p[12])
                crossdustsync2 = p[13]*(func.mbb_uK(nu[i],p[1],p[2])*lognui*func.PL_uK(nu[j],p[4])+ func.PL_uK(nu[i],p[4])*func.mbb_uK(nu[j],p[1],p[2])*lognuj)/(func.PL_uK(nurefs,p[4])*func.mbb_uK(nuref,p[1],p[2]))
                crossdustsync3 = p[14]*(func.mbb_uK(nu[i],p[1],p[2])*(dxi-dx0)*func.PL_uK(nu[j],p[4])+ func.PL_uK(nu[i],p[4])*func.mbb_uK(nu[j],p[1],p[2])*(dxj-dx0))/(func.PL_uK(nurefs,p[4])*func.mbb_uK(nuref,p[1],p[2]))
                crossdustsync4 = p[15]*(func.mbb_uK(nu[i],p[1],p[2])*lognujs*func.PL_uK(nu[j],p[4])+ func.PL_uK(nu[i],p[4])*func.mbb_uK(nu[j],p[1],p[2])*lognuis)/(func.PL_uK(nurefs,p[4])*func.mbb_uK(nuref,p[1],p[2]))
                crossdustsync5 = p[16]*(func.mbb_uK(nu[i],p[1],p[2])*lognui*func.PL_uK(nu[j],p[4])*lognujs+ func.PL_uK(nu[i],p[4])*lognuis*func.mbb_uK(nu[j],p[1],p[2])*lognuj)/(func.PL_uK(nurefs,p[4])*func.mbb_uK(nuref,p[1],p[2]))
                crossdustsync6 = p[17]*(func.mbb_uK(nu[i],p[1],p[2])*(dxi-dx0)*func.PL_uK(nu[j],p[4])*lognujs+ func.PL_uK(nu[i],p[4])*lognuis*func.mbb_uK(nu[j],p[1],p[2])*(dxj-dx0))/(func.PL_uK(nurefs,p[4])*func.mbb_uK(nuref,p[1],p[2]))
                model[icross] = temp + temp2 + syncmom+ crossdustsync+ crossdustsync2+ crossdustsync3+crossdustsync4+crossdustsync5+crossdustsync6+ DL_lensbin[int(p[19])] + p[18]*DL_tens[int(p[19])]
                icross = icross + 1
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

def Fitdcbeta2T_PL(p,fjac=None, x=None, y=None, err=None):
    nuref  = 353
    nurefs=23.
    ncross = len(x)
    nnus   = int((-1 + np.sqrt(ncross*8+1))/2.)
    posauto = [int(nnus*i - i*(i+1)/2 + i) for i in range(nnus)]
    nu = x[0:ncross]
    nu = nu[posauto]
    icross = 0
    model = np.zeros(ncross)
    for i in range(nnus):
        for j in range(i,nnus):
                ampl = (func.mbb(nu[i],p[1],p[2])*func.mbb(nu[j],p[1],p[2])/(func.mbb(nuref,p[1],p[2])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                sync= p[3]*(func.PL_uK(nu[i],p[4])*func.PL_uK(nu[j],p[4])/(func.PL_uK(nurefs,p[4])**2))
                crossdustsync= p[5]*(func.mbb_uK(nu[i],p[1],p[2])*func.PL_uK(nu[j],p[4])+ func.PL_uK(nu[i],p[4])*func.mbb_uK(nu[j],p[1],p[2]))/(func.PL_uK(nurefs,p[4])*func.mbb_uK(nuref,p[1],p[2]))
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognui2 = np.log(nui)**2
                lognuj = np.log(nuj)
                lognuj2 = np.log(nuj)**2
                dx0 = func.dmbbT(nuref,p[2])
                dxi = func.dmbbT(nu[i],p[2])
                dxj = func.dmbbT(nu[j],p[2])
                temp = ampl * (p[0]+ (lognui+lognuj) * p[6]+ lognui*lognuj * p[7])
                tempt=ampl*((dxi+dxj-2*dx0)*p[8]+(lognuj*(dxi-dx0)+lognui*(dxj-dx0))*p[9]+(dxi-dx0)*(dxj-dx0)*p[10])
                temp2=ampl*(0.5*(lognui2+lognuj2) *p[11] +0.5 * (lognui2*lognuj+lognui*lognuj2) * p[12]+0.25* (lognui2*lognuj2) * p[13])
                model[icross] = temp + tempt +temp2 + sync + crossdustsync + DL_lensbin[int(p[15])] + p[14]*DL_tens[int(p[15])]
                icross = icross + 1
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

def Fitsc(p,fjac=None, x=None, y=None, err=None):
    nuref  = 353
    nurefs=23.
    ncross = len(x)
    nnus   = int((-1 + np.sqrt(ncross*8+1))/2.)
    posauto = [int(nnus*i - i*(i+1)/2 + i) for i in range(nnus)]
    nu = x[0:ncross]
    nu = nu[posauto]
    icross = 0
    model = np.zeros(ncross)
    for i in range(nnus):
        for j in range(i,nnus):
                sync= p[0]*(func.PL_uK(nu[i],p[1])*func.PL_uK(nu[j],p[1]))
                model[icross] =  sync+ DL_lensbin[int(p[3])] + p[2]*DL_tens[int(p[3])]
                icross = icross + 1
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

def Fitscordre1(p,fjac=None, x=None, y=None, err=None):
    nuref  = 353
    nurefs=23.
    ncross = len(x)
    nnus   = int((-1 + np.sqrt(ncross*8+1))/2.)
    posauto = [int(nnus*i - i*(i+1)/2 + i) for i in range(nnus)]
    nu = x[0:ncross]
    nu = nu[posauto]
    icross = 0
    model = np.zeros(ncross)
    for i in range(nnus):
        for j in range(i,nnus):
                nuis = nu[i]/nurefs
                nujs = nu[j]/nurefs
                lognuis = np.log(nuis)
                lognujs = np.log(nujs)
                sync= (func.PL_uK(nu[i],p[1])*func.PL_uK(nu[j],p[1]))
                syncmom = sync * (p[0]+ (lognuis+lognujs) * p[2]+ lognuis*lognujs * p[3])
                model[icross] = syncmom+ DL_lensbin[int(p[5])] + p[4]*DL_tens[int(p[5])]
                icross = icross + 1
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

def FitdbetaT(p,fjac=None, x=None, y=None, err=None):
    nuref  = 353
    ncross = len(x)
    nnus   = int((-1 + np.sqrt(ncross*8+1))/2.)
    posauto = [int(nnus*i - i*(i+1)/2 + i) for i in range(nnus)]
    nu = x[0:ncross]
    nu = nu[posauto]
    icross = 0
    model = np.zeros(ncross)
    for i in range(nnus):
        for j in range(i,nnus):
                ampl = (func.mbb(nu[i],p[0],p[9])*func.mbb(nu[j],p[0],p[9])/(func.mbb(nuref,p[0],p[9])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognuj = np.log(nuj)
                dx0 = func.dmbbT(nuref,p[9])
                dxi = func.dmbbT(nu[i],p[9])
                dxj = func.dmbbT(nu[j],p[9])
                temp = ampl * (p[1]+ (lognui+lognuj) * p[2]+ lognui*lognuj * p[3])
                temp2=ampl*((dxi+dxj-2*dx0)*p[4]+(lognuj*(dxi-dx0)+lognui*(dxj-dx0))*p[5]+(dxi-dx0)*(dxj-dx0)*p[6])
                model[icross] = temp + temp2 + p[7]*DL_tens[int(p[8])]
                icross = icross + 1
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])
