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

nside = 256
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
            model[icross] =  p[0]*(func.mbb_uK(nu[i],p[1],p[2])*func.mbb_uK(nu[j],p[1],p[2])/(func.mbb_uK(nuref,p[1],p[2])**2.)) + DL_lensbin[int(p[4])]+ p[3]*DL_tens[int(p[4])]
            icross = icross + 1
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

def Fitdcordre0_func(x,p):
    ncross = len(x)
    nnus   = int((-1 + np.sqrt(ncross*8+1))/2.) 
    posauto = [int(nnus*i - i*(i+1)/2 + i) for i in range(nnus)]
    nu = x[posauto]
    nuref=353.
    icross = 0
    model = np.zeros(ncross)
    for i in range(0,nnus):
        for j in range(i,nnus):
            model[icross] =  p[0]*(func.mbb_uK(nu[i],p[1],p[2])*func.mbb_uK(nu[j],p[1],p[2])/(func.mbb_uK(nuref,p[1],p[2])**2.)) + DL_lensbin[int(p[4])]+ p[3]*DL_tens[int(p[4])]
            icross = icross + 1
    return(model)

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

def Fitdcordre1(p,fjac=None, x=None, y=None, err=None):
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
                ampl = (func.mbb_uK(nu[i],p[1],p[2])*func.mbb_uK(nu[j],p[1],p[2])/(func.mbb_uK(nuref,p[1],p[2])**2.))
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognuj = np.log(nuj)
                temp = ampl * (p[0]+ (lognui+lognuj) * p[3]+ lognui*lognuj * p[4])
                model[icross] = temp + DL_lensbin[int(p[6])] + p[5]*DL_tens[int(p[6])]
                icross = icross + 1
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

def Fitdcordre1_func(x,p):
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
                ampl = (func.mbb(nu[i],p[0],p[6])*func.mbb(nu[j],p[0],p[6])/(func.mbb(nuref,p[0],p[6])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognuj = np.log(nuj)
                temp = ampl * (p[1]+ (lognui+lognuj) * p[2]+ lognui*lognuj * p[3])
                model[icross] = temp + DL_lensbin[int(p[5])] + p[4]*DL_tens[int(p[5])]
                icross = icross + 1
    status = 0
    return(model)

def Fitdcordre2(p,fjac=None, x=None, y=None, err=None):
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
                ampl = (func.mbb_uK(nu[i],p[1],p[2])*func.mbb_uK(nu[j],p[1],p[2])/(func.mbb_uK(nuref,p[1],p[2])**2.))
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognui2 = np.log(nui)**2
                lognuj = np.log(nuj)
                lognuj2 = np.log(nuj)**2
                temp = ampl * (p[0]+ (lognui+lognuj) * p[3]+ lognui*lognuj * p[4])
                temp2=ampl*(0.5*(lognui2+lognuj2) *p[5] +0.5 * (lognui2*lognuj+lognui*lognuj2) * p[6]+0.25* (lognui2*lognuj2) * p[7])
                model[icross] = temp + temp2 + DL_lensbin[int(p[9])] + p[8]*DL_tens[int(p[9])]
                icross = icross + 1
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

def Fitdcordre2_func(x,p):
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
                ampl = (func.mbb_uK(nu[i],p[1],p[2])*func.mbb_uK(nu[j],p[1],p[2])/(func.mbb_uK(nuref,p[1],p[2])**2.))
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognui2 = np.log(nui)**2
                lognuj = np.log(nuj)
                lognuj2 = np.log(nuj)**2
                temp = ampl * (p[0]+ (lognui+lognuj) * p[3]+ lognui*lognuj * p[4])
                temp2=ampl*(0.5*(lognui2+lognuj2) *p[5] +0.5 * (lognui2*lognuj+lognui*lognuj2) * p[6]+0.25* (lognui2*lognuj2) * p[7])
                model[icross] = temp + temp2 + DL_lensbin[int(p[9])] + p[8]*DL_tens[int(p[9])]
                icross = icross + 1
    return(model)

def Fitdcordre3(p,fjac=None, x=None, y=None, err=None):
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
                ampl = (func.mbb(nu[i],p[0],p[13])*func.mbb(nu[j],p[0],p[13])/(func.mbb(nuref,p[0],p[13])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognui2 = np.log(nui)**2
                lognui3 = np.log(nui)**3
                lognuj = np.log(nuj)
                lognuj2 = np.log(nuj)**2
                lognuj3 = np.log(nuj)**3
                temp = ampl * (p[1]+ (lognui+lognuj) * p[2]+ lognui*lognuj * p[3])
                temp2=ampl*(0.5*(lognui2+lognuj2) *p[4] +0.5 * (lognui2*lognuj+lognui*lognuj2) * p[5]+0.25* (lognui2*lognuj2) * p[6])
                temp3=ampl*((1./6.)*(lognui3+lognuj3)*p[7]+ (1./6.)*(lognui*lognuj3+lognui3*lognuj)*p[8] + (1./12.)*(lognui2*lognuj3+lognui3*lognuj2)*p[9]+ (1./36.)*lognui3*lognuj3*p[10])
                model[icross] = temp + temp2 + temp3 + DL_lensbin[int(p[12])] + p[11]*DL_tens[int(p[12])]
                icross = icross + 1
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])


def Fitdcordre3_func(x,p):
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
                ampl = (func.mbb(nu[i],p[0],p[13])*func.mbb(nu[j],p[0],p[13])/(func.mbb(nuref,p[0],p[13])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognui2 = np.log(nui)**2
                lognui3 = np.log(nui)**3
                lognuj = np.log(nuj)
                lognuj2 = np.log(nuj)**2
                lognuj3 = np.log(nuj)**3
                temp = ampl * (p[1]+ (lognui+lognuj) * p[2]+ lognui*lognuj * p[3])
                temp2=ampl*(0.5*(lognui2+lognuj2) *p[4] +0.5 * (lognui2*lognuj+lognui*lognuj2) * p[5]+0.25* (lognui2*lognuj2) * p[6])
                temp3=ampl*((1./6.)*(lognui3+lognuj3)*p[7]+ (1./6.)*(lognui*lognuj3+lognui3*lognuj)*p[8] + (1./12.)*(lognui2*lognuj3+lognui3*lognuj2)*p[9]+ (1./36.)*lognui3*lognuj3*p[10])
                model[icross] = temp + temp2 + temp3 + DL_lensbin[int(p[12])] + p[11]*DL_tens[int(p[12])]
                icross = icross + 1
    return(model)


def Fitdcordre4(p,fjac=None, x=None, y=None, err=None):
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
                ampl = (func.mbb(nu[i],p[0],p[18])*func.mbb(nu[j],p[0],p[18])/(func.mbb(nuref,p[0],p[18])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognui2 = np.log(nui)**2
                lognui3 = np.log(nui)**3
                lognui4 = np.log(nui)**4
                lognuj = np.log(nuj)
                lognuj2 = np.log(nuj)**2
                lognuj3 = np.log(nuj)**3
                lognuj4 = np.log(nuj)**4
                temp = ampl * (p[1]+ (lognui+lognuj) * p[2]+ lognui*lognuj * p[3])
                temp2=ampl*(0.5*(lognui2+lognuj2) *p[4] +0.5 * (lognui2*lognuj+lognui*lognuj2) * p[5]+0.25* (lognui2*lognuj2) * p[6])
                temp3=ampl*((1./6.)*(lognui3+lognuj3)*p[7]+ (1./6.)*(lognui*lognuj3+lognui3*lognuj)*p[8] + (1./12.)*(lognui2*lognuj3+lognui3*lognuj2)*p[9]+ (1./36.)*lognui3*lognuj3*p[10])
                temp4 = ampl*((1./24.)*(lognui4+lognuj4)*p[11]+ (1./24.)*(lognui*lognuj4+lognui4*lognuj)*p[12] + (1./48.)*(lognui2*lognuj4+lognui4*lognuj2)*p[13]+(1./148.)*(lognui3*lognuj4+lognui4*lognuj3)*p[14]+ (1./576.)*lognui4*lognuj4*p[15])
                model[icross] = temp + temp2+temp3 + temp4 + DL_lensbin[int(p[17])] + p[16]*DL_tens[int(p[17])]
                icross = icross + 1
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

def Fitdcordre4_func(p,x):
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
                ampl = (func.mbb(nu[i],p[0],p[18])*func.mbb(nu[j],p[0],p[18])/(func.mbb(nuref,p[0],p[18])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognui2 = np.log(nui)**2
                lognui3 = np.log(nui)**3
                lognui4 = np.log(nui)**4
                lognuj = np.log(nuj)
                lognuj2 = np.log(nuj)**2
                lognuj3 = np.log(nuj)**3
                lognuj4 = np.log(nuj)**4
                temp = ampl * (p[1]+ (lognui+lognuj) * p[2]+ lognui*lognuj * p[3])
                temp2=ampl*(0.5*(lognui2+lognuj2) *p[4] +0.5 * (lognui2*lognuj+lognui*lognuj2) * p[5]+0.25* (lognui2*lognuj2) * p[6])
                temp3=ampl*((1./6.)*(lognui3+lognuj3)*p[7]+ (1./6.)*(lognui*lognuj3+lognui3*lognuj)*p[8] + (1./12.)*(lognui2*lognuj3+lognui3*lognuj2)*p[9]+ (1./36.)*lognui3*lognuj3*p[10])
                temp4 = ampl*((1./24.)*(lognui4+lognuj4)*p[11]+ (1./24.)*(lognui*lognuj4+lognui4*lognuj)*p[12] + (1./48.)*(lognui2*lognuj4+lognui4*lognuj2)*p[13]+(1./148.)*(lognui3*lognuj4+lognui4*lognuj3)*p[14]+ (1./576.)*lognui4*lognuj4*p[15])
                model[icross] = temp + temp2+temp3 + temp4 + DL_lensbin[int(p[17])] + p[16]*DL_tens[int(p[17])]
                icross = icross + 1
    return(model)



def FitdcFIXordre2(p,fjac=None, x=None, y=None, err=None):
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
                ampl = (func.mbb(nu[i],p[0],p[7])*func.mbb(nu[j],p[0],p[7])/(func.mbb(nuref,p[0],p[7])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognui2 = np.log(nui)**2
                lognuj = np.log(nuj)
                lognuj2 = np.log(nuj)**2
                temp2=ampl*(p[1]+0.5*(lognui2+lognuj2) *p[2] +0.5 * (lognui2*lognuj+lognui*lognuj2) * p[3]+0.25* (lognui2*lognuj2) * p[4])
                model[icross] =  temp2 + DL_lensbin[int(p[6])] + p[5]*DL_tens[int(p[6])]
                icross = icross + 1
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

def FitdcFIXordre3(p,fjac=None, x=None, y=None, err=None):
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
                ampl = (func.mbb(nu[i],p[0],p[11])*func.mbb(nu[j],p[0],p[11])/(func.mbb(nuref,p[0],p[11])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognui2 = np.log(nui)**2
                lognui3 = np.log(nui)**3
                lognuj = np.log(nuj)
                lognuj2 = np.log(nuj)**2
                lognuj3 = np.log(nuj)**3
                temp2=ampl*(p[1]+0.5*(lognui2+lognuj2) *p[2] +0.5 * (lognui2*lognuj+lognui*lognuj2) * p[3]+0.25* (lognui2*lognuj2) * p[4])
                temp3=ampl*((1./6.)*(lognui3+lognuj3)*p[5]+ (1./6.)*(lognui*lognuj3+lognui3*lognuj)*p[6] + (1./12.)*(lognui2*lognuj3+lognui3*lognuj2)*p[7]+ (1./36.)*lognui3*lognuj3*p[8])
                model[icross] = temp2+temp3 + DL_lensbin[int(p[10])] + p[9]*DL_tens[int(p[10])]
                icross = icross + 1
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])


def FitdcFIXordre4(p,fjac=None, x=None, y=None, err=None):
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
                ampl = (func.mbb(nu[i],p[0],p[16])*func.mbb(nu[j],p[0],p[16])/(func.mbb(nuref,p[0],p[16])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognui2 = np.log(nui)**2
                lognui3 = np.log(nui)**3
                lognui4 = np.log(nui)**4
                lognuj = np.log(nuj)
                lognuj2 = np.log(nuj)**2
                lognuj3 = np.log(nuj)**3
                lognuj4 = np.log(nuj)**4
                temp2=ampl*(p[1]+0.5*(lognui2+lognuj2) *p[2] +0.5 * (lognui2*lognuj+lognui*lognuj2) * p[3]+0.25* (lognui2*lognuj2) * p[4])
                temp3=ampl*((1./6.)*(lognui3+lognuj3)*p[5]+ (1./6.)*(lognui*lognuj3+lognui3*lognuj)*p[6] + (1./12.)*(lognui2*lognuj3+lognui3*lognuj2)*p[7]+ (1./36.)*lognui3*lognuj3*p[8])
                temp4 = ampl*((1./24.)*(lognui4+lognuj4)*p[9]+ (1./24.)*(lognui*lognuj4+lognui4*lognuj)*p[10] + (1./48.)*(lognui2*lognuj4+lognui4*lognuj2)*p[11]+(1./148.)*(lognui3*lognuj4+lognui4*lognuj3)*p[12]+ (1./576.)*lognui4*lognuj4*p[13])
                model[icross] = temp2+temp3 + temp4 + DL_lensbin[int(p[15])] + p[14]*DL_tens[int(p[15])]
                icross = icross + 1
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])


def FitdcSLDordre0(p,fjac=None, x=None, y=None, err=None):
    Nf=9
    Nell=20
    l0 = 0
    Ncross = int(Nf*(Nf+1)/2)
    posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
    nu = x[posauto]
    nuref=353.
    model =  np.zeros(Ncross*Nell)
    for ell in range(Nell):
        icross =0
        for i in range(0,Nf):
            for j in range(i,Nf):
                model[ell*Ncross+icross] =  p[ell]*(func.mbb(nu[i],p[Nell+ell],p[2*Nell+ell])*func.mbb(nu[j],p[Nell+ell],p[2*Nell+ell])/(func.mbb(nuref,p[Nell+ell],p[2*Nell+ell])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2) + DL_lensbin[l0+ell] + p[3*Nell]*DL_tens[l0+ell]
                icross = icross +1
    status = 0
    #return([status, (y-model)/err])
    return([status,np.dot(np.transpose(y-model), err) ])

def FitdcSLDordre0_func(x,p):
    l0 = 2
    Nf=9
    Nell=10
    Ncross = int(Nf*(Nf+1)/2)
    posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
    nu = x[posauto]
    nuref=353.
    model =  np.zeros(Ncross*Nell)
    for ell in range(Nell):
        icross =0
        for i in range(0,Nf):
            for j in range(i,Nf):
                model[ell*Ncross+icross] =  p[ell]*(func.mbb(nu[i],p[Nell+ell],p[2*Nell+ell])*func.mbb(nu[j],p[Nell+ell],p[2*Nell+ell])/(func.mbb(nuref,p[Nell+ell],p[2*Nell+ell])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2) + DL_lensbin[l0+ell] + p[3*Nell]*DL_tens[l0+ell]
                icross = icross + 1
    return(model)

def FitdcSLDordre1(p,fjac=None, x=None, y=None, err=None):
    Nf=9
    Nell=20
    l0 = 0
    Ncross = int(Nf*(Nf+1)/2)
    posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
    nu = x[posauto]
    nuref=353.
    model =  np.zeros(Ncross*Nell)
    for ell in range(Nell):
        icross =0
        for i in range(0,Nf):
            for j in range(i,Nf):
                ampl = (func.mbb(nu[i],p[Nell +ell],p[4*Nell+ell])*func.mbb(nu[j],p[Nell + ell],p[4*Nell+ell])/(func.mbb(nuref,p[Nell+ell],p[4*Nell+ell])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognuj = np.log(nuj)
                temp = ampl * (p[ell]+ (lognui+lognuj) * p[2*Nell+ell]+ lognui*lognuj * p[3*Nell+ell])
                model[ell*Ncross+icross] = temp + DL_lensbin[l0+ell] + p[5*Nell]*DL_tens[l0 +ell]
                icross = icross + 1
    status = 0
    #return([status, (y-model)/err])
    return([status,np.dot(np.transpose(y-model), err) ])


def FitdcSLDordre1_func(x,p):
    Nf=9
    Nell=20
    l0= 0
    Ncross = int(Nf*(Nf+1)/2)
    posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
    nu = x[posauto]
    nuref=353.
    model =  np.zeros(Ncross*Nell)
    for ell in range(Nell):
        icross =0
        for i in range(0,Nf):
            for j in range(i,Nf):
                ampl = (func.mbb(nu[i],p[Nell +ell],p[4*Nell+ell])*func.mbb(nu[j],p[Nell + ell],p[4*Nell+ell])/(func.mbb(nuref,p[Nell+ell],p[4*Nell+ell])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognuj = np.log(nuj)
                temp = ampl * (p[ell]+ (lognui+lognuj) * p[2*Nell+ell]+ lognui*lognuj * p[3*Nell+ell])
                model[ell*Ncross+icross] = temp + DL_lensbin[l0+ell] + p[5*Nell]*DL_tens[l0+ell]
                icross = icross + 1
    return(model)


def FitdcSLDordre2(p,fjac=None, x=None, y=None, err=None):
    Nf=9
    Nell=20
    l0 = 0
    Ncross = int(Nf*(Nf+1)/2)
    posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
    nu = x[posauto]
    nuref=353.
    model =  np.zeros(Ncross*Nell)
    for ell in range(Nell):
        icross =0
        for i in range(0,Nf):
            for j in range(i,Nf):
                ampl = (func.mbb(nu[i],p[Nell+ell],p[7*Nell+ell])*func.mbb(nu[j],p[Nell+ell],p[7*Nell+ell])/(func.mbb(nuref,p[Nell+ell],p[7*Nell+ell])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognui2 = np.log(nui)**2
                lognuj = np.log(nuj)
                lognuj2 = np.log(nuj)**2
                temp = ampl * (p[ell]+ (lognui+lognuj) * p[2*Nell+ell]+ lognui*lognuj * p[3*Nell+ell])
                temp2=ampl*(0.5*(lognui2+lognuj2) *p[4*Nell+ell] +0.5 * (lognui2*lognuj+lognui*lognuj2) * p[5*Nell+ell]+0.25* (lognui2*lognuj2) * p[6*Nell+ell])
                model[ell*Ncross+icross] = temp + temp2 + DL_lensbin[ell+l0] + p[8*Nell]*DL_tens[ell+l0]
                icross = icross + 1
    status = 0
    #return([status, (y-model)/err])
    return([status,np.dot(np.transpose(y-model), err) ])


def FitdcSLDordre2_func(x,p):
    Nf=9
    l0 = 0
    Nell=20
    Ncross = int(Nf*(Nf+1)/2)
    posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
    nu = x[posauto]
    nuref=353.
    model =  np.zeros(Ncross*Nell)
    for ell in range(Nell):
        icross =0
        for i in range(0,Nf):
            for j in range(i,Nf):
                ampl = (func.mbb(nu[i],p[Nell+ell],p[7*Nell+ell])*func.mbb(nu[j],p[Nell+ell],p[7*Nell+ell])/(func.mbb(nuref,p[Nell+ell],p[7*Nell+ell])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognui2 = np.log(nui)**2
                lognuj = np.log(nuj)
                lognuj2 = np.log(nuj)**2
                temp = ampl * (p[ell]+ (lognui+lognuj) * p[2*Nell+ell]+ lognui*lognuj * p[3*Nell+ell])
                temp2=ampl*(0.5*(lognui2+lognuj2) *p[4*Nell+ell] +0.5 * (lognui2*lognuj+lognui*lognuj2) * p[5*Nell+ell]+0.25* (lognui2*lognuj2) * p[6*Nell+ell])
                model[ell*Ncross+icross] = temp + temp2 + DL_lensbin[ell+l0] + p[8*Nell]*DL_tens[ell+l0]
                icross = icross + 1
    return(model)

def FitdcSLDordre3(p,fjac=None, x=None, y=None, err=None):
    Nf=9
    l0=0
    Nell=20
    Ncross = int(Nf*(Nf+1)/2)
    posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
    nu = x[posauto]
    nuref=353.
    model =  np.zeros(Ncross*Nell)
    for ell in range(Nell):
        icross =0
        for i in range(0,Nf):
            for j in range(i,Nf):
                ampl = (func.mbb(nu[i],p[Nell+ell],p[11*Nell+ell])*func.mbb(nu[j],p[Nell+ell],p[11*Nell+ell])/(func.mbb(nuref,p[Nell+ell],p[11*Nell+ell])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognui2 = np.log(nui)**2
                lognui3 = np.log(nui)**3
                lognuj = np.log(nuj)
                lognuj2 = np.log(nuj)**2
                lognuj3 = np.log(nuj)**3
                temp = ampl * (p[ell]+ (lognui+lognuj) * p[2*Nell+ell]+ lognui*lognuj * p[3*Nell+ell])
                temp2=ampl*(0.5*(lognui2+lognuj2) *p[4*Nell+ell] +0.5 * (lognui2*lognuj+lognui*lognuj2) * p[5*Nell+ell]+0.25* (lognui2*lognuj2) * p[6*Nell+ell])
                temp3=ampl*((1./6.)*(lognui3+lognuj3)*p[7*Nell+ell]+ (1./6.)*(lognui*lognuj3+lognui3*lognuj)*p[8*Nell+ell] + (1./12.)*(lognui2*lognuj3+lognui3*lognuj2)*p[9*Nell+ell]+ (1./36.)*lognui3*lognuj3*p[10*Nell+ell])
                model[ell*Ncross+icross] = temp + temp2+temp3 + DL_lensbin[l0+ell] + p[12*Nell]*DL_tens[l0+ell]
                icross = icross + 1
    status = 0
    #return([status, (y-model)/err])
    return([status,np.dot(np.transpose(y-model), err) ])

def FitdcSLDordre3_func(p,fjac=None, x=None, y=None, err=None):
    Nf=9
    l0=0
    Nell=20
    Ncross = int(Nf*(Nf+1)/2)
    posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
    nu = x[posauto]
    nuref=353.
    model =  np.zeros(Ncross*Nell)
    for ell in range(Nell):
        icross =0
        for i in range(0,Nf):
            for j in range(i,Nf):
                ampl = (func.mbb(nu[i],p[Nell+ell],p[11*Nell+ell])*func.mbb(nu[j],p[Nell+ell],p[11*Nell+ell])/(func.mbb(nuref,p[Nell+ell],p[11*Nell+ell])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognui2 = np.log(nui)**2
                lognui3 = np.log(nui)**3
                lognuj = np.log(nuj)
                lognuj2 = np.log(nuj)**2
                lognuj3 = np.log(nuj)**3
                temp = ampl * (p[ell]+ (lognui+lognuj) * p[2*Nell+ell]+ lognui*lognuj * p[3*Nell+ell])
                temp2=ampl*(0.5*(lognui2+lognuj2) *p[4*Nell+ell] +0.5 * (lognui2*lognuj+lognui*lognuj2) * p[5*Nell+ell]+0.25* (lognui2*lognuj2) * p[6*Nell+ell])
                temp3=ampl*((1./6.)*(lognui3+lognuj3)*p[7*Nell+ell]+ (1./6.)*(lognui*lognuj3+lognui3*lognuj)*p[8*Nell+ell] + (1./12.)*(lognui2*lognuj3+lognui3*lognuj2)*p[9*Nell+ell]+ (1./36.)*lognui3*lognuj3*p[10*Nell+ell])
                model[ell*Ncross+icross] = temp + temp2+temp3 + DL_lensbin[l0+ell] + p[12*Nell]*DL_tens[l0+ell]
                icross = icross + 1
    status = 0
    return model


def FitdcSLDordre4(p,fjac=None, x=None, y=None, err=None):
    Nf=9
    Nell=6
    Ncross = int(Nf*(Nf+1)/2)
    posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
    nu = x[posauto]
    nuref=353.
    model =  np.zeros(Ncross*Nell)
    for ell in range(Nell):
        icross =0
        for i in range(0,Nf):
            for j in range(i,Nf):
                ampl = (func.mbb(nu[i],p[Nell+ell],p[16*Nell+ell])*func.mbb(nu[j],p[Nell+ell],p[16*Nell+ell])/(func.mbb(nuref,p[Nell+ell],p[16*Nell+ell])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognui2 = np.log(nui)**2
                lognui3 = np.log(nui)**3
                lognui4 = np.log(nui)**4
                lognuj = np.log(nuj)
                lognuj2 = np.log(nuj)**2
                lognuj3 = np.log(nuj)**3
                lognuj4 = np.log(nuj)**4
                temp = ampl * (p[ell]+ (lognui+lognuj) * p[2*Nell+ell]+ lognui*lognuj * p[3*Nell+ell])
                temp2=ampl*(0.5*(lognui2+lognuj2) *p[4*Nell+ell] +0.5 * (lognui2*lognuj+lognui*lognuj2) * p[5*Nell+ell]+0.25* (lognui2*lognuj2) * p[6*Nell+ell])
                temp3=ampl*((1./6.)*(lognui3+lognuj3)*p[7*Nell+ell]+ (1./6.)*(lognui*lognuj3+lognui3*lognuj)*p[8*Nell+ell] + (1./12.)*(lognui2*lognuj3+lognui3*lognuj2)*p[9*Nell+ell]+ (1./36.)*lognui3*lognuj3*p[10*Nell+ell])
                temp4 = ampl*((1./24.)*(lognui4+lognuj4)*p[11*Nell+ell]+ (1./24.)*(lognui*lognuj4+lognui4*lognuj)*p[12*Nell+ell] + (1./48.)*(lognui2*lognuj4+lognui4*lognuj2)*p[13*Nell+ell]+(1./148.)*(lognui3*lognuj4+lognui4*lognuj3)*p[14*Nell+ell]+ (1./576.)*lognui4*lognuj4*p[15*Nell+ell])
                model[ell*Ncross+icross] = temp + temp2+temp3 + temp4+ DL_lensbin[ell] + p[17*Nell]*DL_tens[ell]
                icross = icross + 1
    status = 0
    #return([status, (y-model)/err])
    return([status,np.dot(np.transpose(y-model), err) ])


def FitdcSLDordre4_func(x,p):
    Nf=9
    Nell=6
    Ncross = int(Nf*(Nf+1)/2)
    posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
    nu = x[posauto]
    nuref=353.
    model =  np.zeros(Ncross*Nell)
    for ell in range(Nell):
        icross =0
        for i in range(0,Nf):
            for j in range(i,Nf):
                ampl = (func.mbb(nu[i],p[Nell+ell],p[16*Nell+ell])*func.mbb(nu[j],p[Nell+ell],p[16*Nell+ell])/(func.mbb(nuref,p[Nell+ell],p[16*Nell+ell])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognui2 = np.log(nui)**2
                lognui3 = np.log(nui)**3
                lognui4 = np.log(nui)**4
                lognuj = np.log(nuj)
                lognuj2 = np.log(nuj)**2
                lognuj3 = np.log(nuj)**3
                lognuj4 = np.log(nuj)**4
                temp = ampl * (p[ell]+ (lognui+lognuj) * p[2*Nell+ell]+ lognui*lognuj * p[3*Nell+ell])
                temp2=ampl*(0.5*(lognui2+lognuj2) *p[4*Nell+ell] +0.5 * (lognui2*lognuj+lognui*lognuj2) * p[5*Nell+ell]+0.25* (lognui2*lognuj2) * p[6*Nell+ell])
                temp3=ampl*((1./6.)*(lognui3+lognuj3)*p[7*Nell+ell]+ (1./6.)*(lognui*lognuj3+lognui3*lognuj)*p[8*Nell+ell] + (1./12.)*(lognui2*lognuj3+lognui3*lognuj2)*p[9*Nell+ell]+ (1./36.)*lognui3*lognuj3*p[10*Nell+ell])
                temp4 = ampl*((1./24.)*(lognui4+lognuj4)*p[11*Nell+ell]+ (1./24.)*(lognui*lognuj4+lognui4*lognuj)*p[12*Nell+ell] + (1./48.)*(lognui2*lognuj4+lognui4*lognuj2)*p[13*Nell+ell]+(1./148.)*(lognui3*lognuj4+lognui4*lognuj3)*p[14*Nell+ell]+ (1./576.)*lognui4*lognuj4*p[15*Nell+ell])
                model[ell*Ncross+icross] = temp + temp2+temp3 + temp4+ DL_lensbin[ell] + p[17*Nell]*DL_tens[ell]
                icross = icross + 1
    status = 0
    #return([status, (y-model)/err])
    return([status,np.dot(np.transpose(y-model), err) ])


def modeld1cordre0_plaw(p,fjac=None, x=None, y=None, err=None):
    ells = np.array([  6.5,  16.5,  26.5,  36.5,  46.5,  56.5,  66.5,  76.5,  86.5,
        96.5, 106.5, 116.5, 126.5, 136.5, 146.5, 156.5, 166.5, 176.5,
       186.5, 196.5])
    nuref  = 353
    Nell = 20
    temp_dust =20
    ncross = len(x)
    nnus   = int((-1 + np.sqrt(ncross*8+1))/2.)
    posauto = [int(nnus*i - i*(i+1)/2 + i) for i in range(nnus)]
    nu = x[0:ncross]
    nu = nu[posauto]
    icross = 0
    model = np.zeros((ncross,Nell))
    for i in range(nnus):
        for j in range(i,nnus):
            nui = nu[i]/nuref
            nuj = nu[j]/nuref
            lognui = np.log(nui)
            lognuj = np.log(nuj)
            ampl = func.mbb(nu[i],p[1],temp_dust)*func.mbb(nu[j],p[1],temp_dust)/(func.mbb(nuref,p[1],temp_dust)**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2 * (ells/6.5)**p[3]
            model[icross] = ampl*(p[0])
            icross = icross + 1
    model += p[2] * DL_tens + DL_lensbin
    model = np.swapaxes(model,0,1)
    model = model.flatten()
    status = 0
    #return([status, (y-model)/err])
    return([status,np.dot(np.transpose(y-model), err) ])


def modeld1cordre1_plaw(p,fjac=None, x=None, y=None, err=None):
    ells = np.array([  6.5,  16.5,  26.5,  36.5,  46.5,  56.5,  66.5,  76.5,  86.5,
        96.5, 106.5, 116.5, 126.5, 136.5, 146.5, 156.5, 166.5, 176.5,
       186.5, 196.5])
    nuref  = 353
    Nell = 20
    temp_dust =21.9023
    ncross = len(x)
    nnus   = int((-1 + np.sqrt(ncross*8+1))/2.)
    posauto = [int(nnus*i - i*(i+1)/2 + i) for i in range(nnus)]
    nu = x[0:ncross]
    nu = nu[posauto]
    icross = 0
    model = np.zeros((ncross,Nell))
    for i in range(nnus):
        for j in range(i,nnus):
            nui = nu[i]/nuref
            nuj = nu[j]/nuref
            lognui = np.log(nui)
            lognuj = np.log(nuj)
            ampl = func.mbb(nu[i],p[1],temp_dust)*func.mbb(nu[j],p[1],temp_dust)/(func.mbb(nuref,p[1],temp_dust)**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2 * (ells/6.5)**p[5]
            model[icross] = ampl*(p[0] + (lognui+lognuj) *  p[2] +  lognui*lognuj  *  p[3])
            icross = icross + 1
    model += p[4] * DL_tens + DL_lensbin
    model = np.swapaxes(model,0,1)
    model = model.flatten()
    status = 0
    #return([status, (y-model)/err])
    return([status,np.dot(np.transpose(y-model), err) ])

def FitdSLDordre0(p,fjac=None, x=None, y=None, err=None):
    Nf=9
    Nell=20
    l0 = 0
    Ncross = int(Nf*(Nf+1)/2)
    posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
    nu = x[posauto]
    nuref=353.
    model =  np.zeros(Ncross*Nell)
    for ell in range(Nell):
        icross =0
        for i in range(0,Nf):
            for j in range(i,Nf):
                model[ell*Ncross+icross] =  p[ell]*(func.mbb(nu[i],p[Nell+ell],p[2*Nell+ell])*func.mbb(nu[j],p[Nell+ell],p[2*Nell+ell])/(func.mbb(nuref,p[Nell+ell],p[2*Nell+ell])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2) +  p[3*Nell]*DL_tens[l0+ell]
                icross = icross +1
    status = 0
    #return([status, (y-model)/err])
    return([status,np.dot(np.transpose(y-model), err) ])

def FitdSLDordre1(p,fjac=None, x=None, y=None, err=None):
    Nf=9
    Nell=20
    l0 = 0
    Ncross = int(Nf*(Nf+1)/2)
    posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
    nu = x[posauto]
    nuref=353.
    model =  np.zeros(Ncross*Nell)
    for ell in range(Nell):
        icross =0
        for i in range(0,Nf):
            for j in range(i,Nf):
                ampl = (func.mbb(nu[i],p[Nell +ell],p[4*Nell+ell])*func.mbb(nu[j],p[Nell + ell],p[4*Nell+ell])/(func.mbb(nuref,p[Nell+ell],p[4*Nell+ell])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognuj = np.log(nuj)
                temp = ampl * (p[ell]+ (lognui+lognuj) * p[2*Nell+ell]+ lognui*lognuj * p[3*Nell+ell])
                model[ell*Ncross+icross] = temp + p[5*Nell]*DL_tens[l0 +ell]
                icross = icross + 1
    status = 0
    #return([status, (y-model)/err])
    return([status,np.dot(np.transpose(y-model), err) ])


def FitdSLDordre2(p,fjac=None, x=None, y=None, err=None):
    Nf=9
    Nell=20
    l0 = 0
    Ncross = int(Nf*(Nf+1)/2)
    posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
    nu = x[posauto]
    nuref=353.
    model =  np.zeros(Ncross*Nell)
    for ell in range(Nell):
        icross =0
        for i in range(0,Nf):
            for j in range(i,Nf):
                ampl = (func.mbb(nu[i],p[Nell+ell],p[7*Nell+ell])*func.mbb(nu[j],p[Nell+ell],p[7*Nell+ell])/(func.mbb(nuref,p[Nell+ell],p[7*Nell+ell])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognui2 = np.log(nui)**2
                lognuj = np.log(nuj)
                lognuj2 = np.log(nuj)**2
                temp = ampl * (p[ell]+ (lognui+lognuj) * p[2*Nell+ell]+ lognui*lognuj * p[3*Nell+ell])
                temp2=ampl*(0.5*(lognui2+lognuj2) *p[4*Nell+ell] +0.5 * (lognui2*lognuj+lognui*lognuj2) * p[5*Nell+ell]+0.25* (lognui2*lognuj2) * p[6*Nell+ell])
                model[ell*Ncross+icross] = temp + temp2  + p[8*Nell]*DL_tens[ell+l0]
                icross = icross + 1
    status = 0
    #return([status, (y-model)/err])
    return([status,np.dot(np.transpose(y-model), err) ])

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
                ampl = (func.mbb_uK(nu[i],p[1],p[2])*func.mbb_uK(nu[j],p[1],p[2])/(func.mbb_uK(nuref,p[1],p[2])**2.))
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
                ampl = (func.mbb_uK(nu[i],p[1],p[2])*func.mbb_uK(nu[j],p[1],p[2])/(func.mbb_uK(nuref,p[1],p[2])**2.))
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
                ampl = (func.mbb_uK(nu[i],p[1],p[2])*func.mbb_uK(nu[j],p[1],p[2])/(func.mbb_uK(nuref,p[1],p[2])**2.))
                sync= p[3]*(func.PL_uK(nu[i],p[4])*func.PL_uK(nu[j],p[4])/(func.PL_uK(nurefs,p[4])**2))
                crossdustsync= (func.mbb_uK(nu[i],p[1],p[2])*func.PL_uK(nu[j],p[4])+ func.PL_uK(nu[i],p[4])*func.mbb_uK(nu[j],p[1],p[2]))/(func.PL_uK(nurefs,p[4])*func.mbb_uK(nuref,p[1],p[2]))
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
                temp2=ampl*((dxi+dxj-2*dx0)*p[8]+(lognuj*(dxi-dx0)+lognui*(dxj-dx0))*p[9]+(dxi-dx0)*(dxj-dx0)*p[10])
                syncmom = sync * (p[5]+ (lognuis+lognujs) * p[11]+ lognuis*lognujs * p[12])
                model[icross] = temp + temp2 + syncmom+crossdustsync+ DL_lensbin[int(p[14])] + p[13]*DL_tens[int(p[14])]
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
                sync= p[0]*(func.PL_uK(nu[i],p[1])*func.PL_uK(nu[j],p[1])/(func.PL_uK(nurefs,p[1])**2))
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
                sync= (func.PL_uK(nu[i],p[1])*func.PL_uK(nu[j],p[1])/(func.PL_uK(nurefs,p[1])**2))
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

def FitdcSLDbetaT(p,fjac=None, x=None, y=None, err=None):
    Nf=9
    Nell=20
    l0 = 0
    Ncross = int(Nf*(Nf+1)/2)
    posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
    nu = x[posauto]
    nuref=353.
    model =  np.zeros(Ncross*Nell)
    for ell in range(Nell):
        icross =0
        for i in range(0,Nf):
            for j in range(i,Nf):
                ampl = (func.mbb(nu[i],p[Nell+ell],p[7*Nell+ell])*func.mbb(nu[j],p[Nell+ell],p[7*Nell+ell])/(func.mbb(nuref,p[Nell+ell],p[7*Nell+ell])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognuj = np.log(nuj)
                dx0 = func.dmbbT(nuref,p[7*Nell+ell])
                dxi = func.dmbbT(nu[i],p[7*Nell+ell])
                dxj = func.dmbbT(nu[j],p[7*Nell+ell])
                temp = ampl * (p[ell]+ (lognui+lognuj) * p[2*Nell+ell]+ lognui*lognuj * p[3*Nell+ell])
                temp2=ampl*((dxi+dxj-2*dx0)*p[4*Nell+ell]+(lognuj*(dxi-dx0)+lognui*(dxj-dx0))*p[5*Nell+ell]+(dxi-dx0)*(dxj-dx0)*p[6*Nell+ell])
                model[ell*Ncross+icross] = temp + temp2 + DL_lensbin[ell+l0] + p[8*Nell]*DL_tens[ell+l0]
                icross = icross + 1
    status = 0
    #return([status, (y-model)/err])
    return([status,np.dot(np.transpose(y-model), err) ])

def FitdSLDbetaT(p,fjac=None, x=None, y=None, err=None):
    Nf=9
    Nell=20
    l0 = 0
    Ncross = int(Nf*(Nf+1)/2)
    posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
    nu = x[posauto]
    nuref=353.
    model =  np.zeros(Ncross*Nell)
    for ell in range(Nell):
        icross =0
        for i in range(0,Nf):
            for j in range(i,Nf):
                ampl = (func.mbb(nu[i],p[Nell+ell],p[7*Nell+ell])*func.mbb(nu[j],p[Nell+ell],p[7*Nell+ell])/(func.mbb(nuref,p[Nell+ell],p[7*Nell+ell])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognuj = np.log(nuj)
                dx0 = func.dmbbT(nuref,p[7*Nell+ell])
                dxi = func.dmbbT(nu[i],p[7*Nell+ell])
                dxj = func.dmbbT(nu[j],p[7*Nell+ell])
                temp = ampl * (p[ell]+ (lognui+lognuj) * p[2*Nell+ell]+ lognui*lognuj * p[3*Nell+ell])
                temp2=ampl*((dxi+dxj-2*dx0)*p[4*Nell+ell]+(lognuj*(dxi-dx0)+lognui*(dxj-dx0))*p[5*Nell+ell]+(dxi-dx0)*(dxj-dx0)*p[6*Nell+ell])
                model[ell*Ncross+icross] = temp + temp2 + p[8*Nell]*DL_tens[ell+l0]
                icross = icross + 1
    status = 0
    #return([status, (y-model)/err])
    return([status,np.dot(np.transpose(y-model), err) ])


def Fitdcbeta2T(p,fjac=None, x=None, y=None, err=None):
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
                ampl = (func.mbb(nu[i],p[0],p[12])*func.mbb(nu[j],p[0],p[12])/(func.mbb(nuref,p[0],p[12])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognui2 = np.log(nui)**2
                lognuj = np.log(nuj)
                lognuj2 = np.log(nuj)**2
                dx0 = func.dmbbT(nuref,p[12])
                dxi = func.dmbbT(nu[i],p[12])
                dxj = func.dmbbT(nu[j],p[12])
                temp = ampl * (p[1]+ (lognui+lognuj) * p[2]+ lognui*lognuj * p[3])
                tempt=ampl*((dxi+dxj-2*dx0)*p[4]+(lognuj*(dxi-dx0)+lognui*(dxj-dx0))*p[5]+(dxi-dx0)*(dxj-dx0)*p[6])
                temp2=ampl*(0.5*(lognui2+lognuj2) *p[7] +0.5 * (lognui2*lognuj+lognui*lognuj2) * p[8]+0.25* (lognui2*lognuj2) * p[9])
                model[icross] = temp + tempt +temp2 + DL_lensbin[int(p[11])] + p[10]*DL_tens[int(p[11])]
                icross = icross + 1
    status = 0
    return([status, np.dot(np.transpose(y-model), err)])

# def Fitd_QB_ordre0(p,fjac=None, x=None, y=None, err=None,Nf):
#     Ncross = int(Nf*(Nf+1)/2)
#
#     posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
#     nu = x[posauto]
#     nuref=353.
#     icross = 0
#     model = np.zeros(Ncross)
#     for i in range(0,Nf):
#         for j in range(i,Nf):
#             model[icross] =  p[0]*(func.mbb(nu[i],p[1],p[4])*func.mbb(nu[j],p[1],p[4])/(func.mbb(nuref,p[1],p[4])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2) + p[2]*DL_tens[int(p[3])]
#             icross = icross + 1
#     status = 0
#     return([status, np.dot(np.transpose(y-model), err)])

def FitsSLDordre0(p,fjac=None, x=None, y=None, err=None):
    Nf=9
    Nell=20
    l0 = 0
    Ncross = int(Nf*(Nf+1)/2)
    posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
    nu = x[posauto]
    nuref=353.
    model =  np.zeros(Ncross*Nell)
    for ell in range(Nell):
        icross =0
        for i in range(0,Nf):
            for j in range(i,Nf):
                sync=  p[0]*((nu[i]*nu[j]/nuref/nuref)**p[1])*(l[ell]**p[2])*psm.convert_units('uK_RJ','uK_CMB',nu[i])*psm.convert_units('uK_RJ','uK_CMB',nu[j])/psm.convert_units('uK_RJ','uK_CMB',nuref)**2 
                model[ell*Ncross+icross] = p[3]*DL_tens[l0+ell] + sync
                icross = icross +1
    status = 0
    #return([status, (y-model)/err])
    return([status,np.dot(np.transpose(y-model), err) ])

def FitdscSLDordre0(p,fjac=None, x=None, y=None, err=None):
    Nell=20
    l0 = 0
    Ncross=int(len(x)/Nell)
    Nf = int((-1 + np.sqrt(Ncross*8+1))/2.)    
    posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
    nu = x[posauto]
    nuref=353.
    nurefs=23.
    model =  np.zeros(Ncross*Nell)
    for ell in range(Nell):
        icross =0
        for i in range(0,Nf):
            for j in range(i,Nf):
                temp = p[ell]*(func.mbb(nu[i],p[Nell+ell],p[2*Nell+ell])*func.mbb(nu[j],p[Nell+ell],p[2*Nell+ell])/(func.mbb(nuref,p[Nell+ell],p[2*Nell+ell])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2) 
                sync=  p[3*Nell]*((nu[i]*nu[j]/nurefs/nurefs)**p[3*Nell+1])*(l[ell]**p[3*Nell+2])*psm.convert_units('uK_RJ','uK_CMB',nu[i])*psm.convert_units('uK_RJ','uK_CMB',nu[j])/psm.convert_units('uK_RJ','uK_CMB',nurefs)**2 
                model[ell*Ncross+icross] =  temp + p[3*Nell+3]*DL_tens[l0+ell] + sync + DL_lensbin[l0+ell] 
                icross = icross +1
    status = 0
    #return([status, (y-model)/err])
    return([status,np.dot(np.transpose(y-model), err) ])

def FitdscSLDordre1(p,fjac=None, x=None, y=None, err=None):
    Nell=20
    l0 = 0
    Ncross=int(len(x)/Nell)
    Nf = int((-1 + np.sqrt(Ncross*8+1))/2.)    
    posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
    nu = x[posauto]
    nuref=353.
    nurefs=23.
    model =  np.zeros(Ncross*Nell)
    for ell in range(Nell):
        icross =0
        for i in range(0,Nf):
            for j in range(i,Nf):
                ampl = (func.mbb(nu[i],p[Nell +ell],p[4*Nell+ell])*func.mbb(nu[j],p[Nell + ell],p[4*Nell+ell])/(func.mbb(nuref,p[Nell+ell],p[4*Nell+ell])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognuj = np.log(nuj)
                sync=  p[5*Nell]*((nu[i]*nu[j]/nurefs/nurefs)**p[5*Nell+1])*(l[ell]**p[5*Nell+2])*psm.convert_units('uK_RJ','uK_CMB',nu[i])*psm.convert_units('uK_RJ','uK_CMB',nu[j])/psm.convert_units('uK_RJ','uK_CMB',nurefs)**2 
                temp = ampl * (p[ell]+ (lognui+lognuj) * p[2*Nell+ell]+ lognui*lognuj * p[3*Nell+ell])
                model[ell*Ncross+icross] = temp + DL_lensbin[l0+ell] + p[5*Nell+3]*DL_tens[l0 +ell] + sync
                icross = icross + 1
    status = 0
    #return([status, (y-model)/err])
    return([status,np.dot(np.transpose(y-model), err) ])

def FitdscSLDordre2(p,fjac=None, x=None, y=None, err=None):
    Nell=20
    l0 = 0
    Ncross=int(len(x)/Nell)
    Nf = int((-1 + np.sqrt(Ncross*8+1))/2.)    
    posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
    nu = x[posauto]
    nuref=353.
    nurefs=23.
    model =  np.zeros(Ncross*Nell)
    for ell in range(Nell):
        icross =0
        for i in range(0,Nf):
            for j in range(i,Nf):
                ampl = (func.mbb(nu[i],p[Nell+ell],p[7*Nell+ell])*func.mbb(nu[j],p[Nell+ell],p[7*Nell+ell])/(func.mbb(nuref,p[Nell+ell],p[7*Nell+ell])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognui2 = np.log(nui)**2
                lognuj = np.log(nuj)
                lognuj2 = np.log(nuj)**2
                sync=  p[3*Nell]*((nu[i]*nu[j]/nurefs/nurefs)**p[8*Nell+1])*(l[ell]**p[8*Nell+2])*psm.convert_units('uK_RJ','uK_CMB',nu[i])*psm.convert_units('uK_RJ','uK_CMB',nu[j])/psm.convert_units('uK_RJ','uK_CMB',nurefs)**2 
                temp = ampl * (p[ell]+ (lognui+lognuj) * p[2*Nell+ell]+ lognui*lognuj * p[3*Nell+ell])
                temp2=ampl*(0.5*(lognui2+lognuj2) *p[4*Nell+ell] +0.5 * (lognui2*lognuj+lognui*lognuj2) * p[5*Nell+ell]+0.25* (lognui2*lognuj2) * p[6*Nell+ell])
                model[ell*Ncross+icross] = temp + temp2 + DL_lensbin[ell+l0] + p[8*Nell+3]*DL_tens[ell+l0] + sync
                icross = icross + 1
    status = 0
    #return([status, (y-model)/err])
    return([status,np.dot(np.transpose(y-model), err) ])


def FitdscSLDbetaT(p,fjac=None, x=None, y=None, err=None):
    Nell=20
    l0 = 0
    Ncross=int(len(x)/Nell)
    Nf = int((-1 + np.sqrt(Ncross*8+1))/2.)    
    posauto = [int(Nf*i - i*(i+1)/2 + i) for i in range(Nf)]
    nu = x[posauto]
    nuref=353.
    nurefs=23.
    model =  np.zeros(Ncross*Nell)
    for ell in range(Nell):
        icross =0
        for i in range(0,Nf):
            for j in range(i,Nf):
                ampl = (func.mbb(nu[i],p[Nell+ell],p[7*Nell+ell])*func.mbb(nu[j],p[Nell+ell],p[7*Nell+ell])/(func.mbb(nuref,p[Nell+ell],p[7*Nell+ell])**2.)*psm.convert_units('MJysr','uK_CMB',nu[i])*psm.convert_units('MJysr','uK_CMB',nu[j])/psm.convert_units('MJysr','uK_CMB',nuref)**2)
                nui = nu[i]/nuref
                nuj = nu[j]/nuref
                lognui = np.log(nui)
                lognuj = np.log(nuj)
                dx0 = func.dmbbT(nuref,p[7*Nell+ell])
                dxi = func.dmbbT(nu[i],p[7*Nell+ell])
                dxj = func.dmbbT(nu[j],p[7*Nell+ell])
                temp = ampl * (p[ell]+ (lognui+lognuj) * p[2*Nell+ell]+ lognui*lognuj * p[3*Nell+ell])
                sync=  p[8*Nell]*((nu[i]*nu[j]/nurefs/nurefs)**p[8*Nell+1])*(l[ell]**p[8*Nell+2])*psm.convert_units('uK_RJ','uK_CMB',nu[i])*psm.convert_units('uK_RJ','uK_CMB',nu[j])/psm.convert_units('uK_RJ','uK_CMB',nurefs)**2 
                temp2=ampl*((dxi+dxj-2*dx0)*p[4*Nell+ell]+(lognuj*(dxi-dx0)+lognui*(dxj-dx0))*p[5*Nell+ell]+(dxi-dx0)*(dxj-dx0)*p[6*Nell+ell])
                model[ell*Ncross+icross] = temp + temp2 + sync+ DL_lensbin[ell+l0] + p[8*Nell+3]*DL_tens[ell+l0] 
                icross = icross + 1
    status = 0
    #return([status, (y-model)/err])
    return([status,np.dot(np.transpose(y-model), err) ])
