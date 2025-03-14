import numpy as np
import basicfunc as func
import pysm3
import healpy as hp
import pymaster as nmt
import numpy as np
import pysm_common as psm 
import basicfunc as func

#contains all the models to be fitted by mpfit 

def getDL_cmb(nside=64,Nlbin=10,mode='BB'):
    lmax = nside*2-1
    b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
    l = b.get_effective_ells()
    CLcmb_or=hp.read_cl('./power_spectra/Cls_Planck2018_r0.fits') #TT EE BB TE
    CL_tens=hp.read_cl('./power_spectra/Cls_Planck2018_tensor_r1.fits')
    sp_dict = {'TT': 0, 'EE': 1, 'BB': 2, 'TE': 3}
    sp = sp_dict.get(mode, None)
    if sp is None:
        raise ValueError("Unknown mode")    
    DL_lensbin = l*(l+1)*b.bin_cell(CLcmb_or[sp,:lmax+1])/2/np.pi
    DL_tens = l*(l+1)*b.bin_cell(CL_tens[sp,:lmax+1])/2/np.pi
    return DL_lensbin, DL_tens

#Likelihoods##########################################################

def lkl_mpfit(p, fjac=None, y=None, err=None, model_func=None, **kwargs):
    '''
    Gaussian likelihood for mpfit
    '''
    if model_func is None:
        raise ValueError("A model must be specified")

    model = model_func(p, **kwargs)
    status = 0
    return [status, np.dot(np.transpose(y - model), err)]

def lnlike(p, y, invcov, model_func=None, **kwargs):
    """Log-likelihood function for emcee"""
    
    if model_func is None:
        raise ValueError("A model must be specified")

    model = model_func(p, **kwargs)

    if np.any(np.isnan(model)) or np.any(np.isinf(model)):
        return -np.inf  

    chi2 = np.dot(np.dot(y - model, invcov), y - model)

    if np.isnan(chi2) or np.isinf(chi2):
        return -np.inf  
    return -0.5 * np.sum(chi2)

def lnprior(p,all_ell=True):
    """Prior function to prevent extreme values in MCMC"""
    if all_ell==True:
        A_d = p[:15]     
        A_s = p[15:30]   
        A_sd = p[30:45]  
        beta_d, T_d_inv, beta_s, r = p[45:]  
        if np.any(A_d < 0) or np.any(A_d > 1000):
            return -np.inf
        if np.any(A_s < 0) or np.any(A_s > 1000):
            return -np.inf
        if np.any(A_sd < -5) or np.any(A_sd > 5):
            return -np.inf
        if not (1 < beta_d < 2):
            return -np.inf
        if not (0 < T_d_inv < 1): 
            return -np.inf
        if not (-5 < beta_s < -1): 
            return -np.inf
        if not (-0.5 < r < 0.5): 
            return -np.inf
        return 0  
    elif all_ell==False:
        A, beta_d, T_d_inv, A_s, beta_s, A_sd, r = p
        if not (0 < A < 1000 and 0 < A_s < 1000 and -5 < A_sd < 5):  # Exemples de bornes
            return -np.inf
        if not (1 < beta_d < 2 and 0 < T_d_inv < 1 and -5 < beta_s < -1 and -0.5 < r < 0.5):
            return -np.inf
        return 0 

def lnprob(p, y, invcov, model_func=None, all_ell=True, **kwargs):
    """Log-probability function for emcee"""

    lp = lnprior(p,all_ell)
    if not np.isfinite(lp):
        return -np.inf  # Exclut les valeurs extrêmes

    return lp + lnlike(p, y, invcov, model_func=model_func, **kwargs)

#Models##########################################################

def Gaussian(p,fjac=None, x=None, y=None, err=None):
    # Gaussian curve with mpfit
    model = func.Gaussian(x,p[0],p[1])
    status = 0
    return([status, (y-model)/err])

def PL_ell(ell,alpha):
    l0=80
    return (ell/l0)**alpha

def func_d_o0(p, x1=None, x2=None,nuref=353.,DL_lensbin=None, DL_tens=None):
    #fit function dust, order 0
    nu_i=x1
    nu_j=x2
    mbb = p[0] * func.mbb_uK(nu_i, p[1], p[2]) * func.mbb_uK(nu_j, p[1], p[2])
    model = mbb  + DL_lensbin[int(p[4])] + p[3] * DL_tens[int(p[4])]
    return model

def func_ds_o0(p, x1=None, x2=None,nuref=353.,nurefs=23.,ell=None,DL_lensbin=None, DL_tens=None):
    #fit function dust+syncrotron, order 0
    nu_i=x1
    nu_j=x2
    mbb = p[0]*func.mbb_uK(nu_i, p[1], p[2],nu0=nuref) * func.mbb_uK(nu_j, p[1], p[2],nu0=nuref)
    sync = p[3]*func.PL_uK(nu_i, p[4],nu0=nurefs) * func.PL_uK(nu_j, p[4],nu0=nurefs)
    normcorr= np.sqrt(abs(p[0]*p[3]))
    #normcorr= 1
    crossdustsync = p[5]*normcorr*(func.mbb_uK(nu_i, p[1], p[2],nu0=nuref) * func.PL_uK(nu_j, p[4],nu0=nurefs) + func.PL_uK(nu_i, p[4],nu0=nurefs) * func.mbb_uK(nu_j, p[1], p[2],nu0=nuref))
    model = mbb + sync + crossdustsync + DL_lensbin[ell] + p[6] * DL_tens[ell]
    return model

def func_ds_o1bt(p, x1=None, x2=None,nuref=353,nurefs=23.,ell=None,DL_lensbin=None, DL_tens=None):
    #fit function dust+syncrotron, order 1 in beta and 1/T
    nu_i=x1
    nu_j=x2
    ampl = func.mbb_uK(nu_i,p[1],p[2],nu0=nuref)*func.mbb_uK(nu_j,p[1],p[2],nu0=nuref)
    sync= p[3]*func.PL_uK(nu_i,p[4],nu0=nurefs)*func.PL_uK(nu_j,p[4],nu0=nurefs)
    normcorr= np.sqrt(abs(p[0]*p[3]))
    #normcorr= 1
    crossdustsync= p[5]*normcorr*(func.mbb_uK(nu_i,p[1],p[2],nu0=nuref)*func.PL_uK(nu_j,p[4],nu0=nurefs)+ func.PL_uK(nu_i,p[4],nu0=nurefs)*func.mbb_uK(nu_j,p[1],p[2],nu0=nuref))
    lognui = np.log(nu_i/nuref)
    lognuj = np.log(nu_j/nuref)
    dx0 = func.dmbb_bT(nuref,p[2])
    dxi = func.dmbb_bT(nu_i,p[2])
    dxj = func.dmbb_bT(nu_j,p[2])
    temp = ampl*(p[0]+ (lognui+lognuj) * p[6]+ lognui*lognuj * p[7])
    temp2= ampl*((dxi+dxj-2*dx0)*p[8]+(lognuj*(dxi-dx0)+lognui*(dxj-dx0))*p[9]+(dxi-dx0)*(dxj-dx0)*p[10])
    crossdustsync2 = p[11]*(func.mbb_uK(nu_i,p[1],p[2],nu0=nuref)*lognui*func.PL_uK(nu_j,p[4],nu0=nurefs)+ func.PL_uK(nu_i,p[4],nu0=nurefs)*func.mbb_uK(nu_j,p[1],p[2],nu0=nuref)*lognuj)
    crossdustsync3 = p[12]*(func.mbb_uK(nu_i,p[1],p[2],nu0=nuref)*(dxi-dx0)*func.PL_uK(nu_j,p[4],nu0=nurefs)+ func.PL_uK(nu_i,p[4],nu0=nurefs)*func.mbb_uK(nu_j,p[1],p[2],nu0=nuref)*(dxj-dx0))
    model = temp + temp2 + sync+ crossdustsync+ crossdustsync2+ crossdustsync3+ DL_lensbin[ell] + p[13]*DL_tens[ell]
    return model

def func_ds_o1bts(p, x1=None, x2=None,nuref=353,nurefs=23.,ell=None,DL_lensbin=None, DL_tens=None):
    #fit function dust+syncrotron, order 1 in beta, beta_s and 1/T
    nu_i=x1
    nu_j=x2
    ampl = func.mbb_uK(nu_i,p[1],p[2],nu0=nuref)*func.mbb_uK(nu_j,p[1],p[2],nu0=nuref)
    sync= func.PL_uK(nu_i,p[4],nu0=nurefs)*func.PL_uK(nu_j,p[4],nu0=nurefs)
    normcorr= np.sqrt(abs(p[0]*p[3]))
    #normcorr= 1
    crossdustsync= p[5]*normcorr*(func.mbb_uK(nu_i,p[1],p[2],nu0=nuref)*func.PL_uK(nu_j,p[4],nu0=nurefs)+ func.PL_uK(nu_i,p[4],nu0=nurefs)*func.mbb_uK(nu_j,p[1],p[2],nu0=nuref))
    lognui = np.log(nu_i/nuref)
    lognuj = np.log(nu_j/nuref)
    lognuis = np.log(nu_i/nurefs)
    lognujs = np.log(nu_j/nurefs)
    dx0 = func.dmbb_bT(nuref,p[2])
    dxi = func.dmbb_bT(nu_i,p[2])
    dxj = func.dmbb_bT(nu_j,p[2])
    temp = ampl * (p[0]+ (lognui+lognuj) * p[6]+ lognui*lognuj * p[7])
    temp2 = ampl*((dxi+dxj-2*dx0)*p[8]+(lognuj*(dxi-dx0)+lognui*(dxj-dx0))*p[9]+(dxi-dx0)*(dxj-dx0)*p[10])
    syncmom = sync * (p[3]+ (lognuis+lognujs) * p[11]+ lognuis*lognujs * p[12])
    crossdustsync2 = p[13]*(func.mbb_uK(nu_i,p[1],p[2],nu0=nuref)*lognui*func.PL_uK(nu_j,p[4],nu0=nurefs)+ func.PL_uK(nu_i,p[4],nu0=nurefs)*func.mbb_uK(nu_j,p[1],p[2],nu0=nuref)*lognuj)
    crossdustsync3 = p[14]*(func.mbb_uK(nu_i,p[1],p[2],nu0=nuref)*(dxi-dx0)*func.PL_uK(nu_j,p[4],nu0=nurefs)+ func.PL_uK(nu_i,p[4],nu0=nurefs)*func.mbb_uK(nu_j,p[1],p[2],nu0=nuref)*(dxj-dx0))
    crossdustsync4 = p[15]*(func.mbb_uK(nu_i,p[1],p[2],nu0=nuref)*lognujs*func.PL_uK(nu_j,p[4],nu0=nurefs)+ func.PL_uK(nu_i,p[4],nu0=nurefs)*func.mbb_uK(nu_j,p[1],p[2],nu0=nuref)*lognuis)
    crossdustsync5 = p[16]*(func.mbb_uK(nu_i,p[1],p[2],nu0=nuref)*lognui*func.PL_uK(nu_j,p[4],nu0=nurefs)*lognujs+ func.PL_uK(nu_i,p[4],nu0=nurefs)*lognuis*func.mbb_uK(nu_j,p[1],p[2],nu0=nuref)*lognuj)
    crossdustsync6 = p[17]*(func.mbb_uK(nu_i,p[1],p[2],nu0=nuref)*(dxi-dx0)*func.PL_uK(nu_j,p[4],nu0=nurefs)*lognujs+ func.PL_uK(nu_i,p[4],nu0=nurefs)*lognuis*func.mbb_uK(nu_j,p[1],p[2],nu0=nuref)*(dxj-dx0))
    model = temp + temp2 + syncmom + crossdustsync+ crossdustsync2+ crossdustsync3+crossdustsync4+crossdustsync5+crossdustsync6+ DL_lensbin[ell] + p[18]*DL_tens[ell]
    return model

#all_ell

def func_ds_o0_all_ell(p, x1=None, x2=None,nuref=353.,nurefs=23.,Nell=None,DL_lensbin=None, DL_tens=None):
    #fit function dust+syncrotron, order 0
    nu_i=x1
    nu_j=x2
    ellim=3*Nell-1
    Ncross=len(nu_i)/Nell
    mbb = np.repeat(p[:Nell],Ncross)*func.mbb_uK(nu_i, p[ellim+1], p[ellim+2],nu0=nuref) * func.mbb_uK(nu_j, p[ellim+1], p[ellim+2],nu0=nuref)
    sync = np.repeat(p[Nell:2*Nell],Ncross)*func.PL_uK(nu_i, p[ellim+3],nu0=nurefs) * func.PL_uK(nu_j, p[ellim+3],nu0=nurefs)
    normcorr= np.repeat(np.sqrt(abs(p[:Nell]*p[Nell:2*Nell])),Ncross)
    #normcorr= 1
    crossdustsync = np.repeat(p[2*Nell:3*Nell],Ncross)*normcorr*(func.mbb_uK(nu_i, p[ellim+1], p[ellim+2],nu0=nuref) * func.PL_uK(nu_j, p[ellim+3],nu0=nurefs) + func.PL_uK(nu_i, p[ellim+3],nu0=nurefs) * func.mbb_uK(nu_j, p[ellim+1], p[ellim+2],nu0=nuref))
    model = mbb + sync + crossdustsync + DL_lensbin + p[ellim+4] * DL_tens
    return model

def func_ds_o1bt_all_ell(p, x1=None, x2=None,nuref=353,nurefs=23.,ell=None,Nell=None,DL_lensbin=None, DL_tens=None):
    #fit function dust+syncrotron, order 1 in beta and T
    nu_i=x1
    nu_j=x2
    ellim=3*Nell-1
    Ncross=len(nu_i)/Nell
    ampl = func.mbb_uK(nu_i, p[ellim+1], p[ellim+2],nu0=nuref) * func.mbb_uK(nu_j, p[ellim+1], p[ellim+2],nu0=nuref)
    sync = np.repeat(p[Nell:2*Nell],Ncross)*func.PL_uK(nu_i, p[ellim+3],nu0=nurefs) * func.PL_uK(nu_j, p[ellim+3],nu0=nurefs)
    normcorr= np.repeat(np.sqrt(abs(p[:Nell]*p[Nell:2*Nell])),Ncross)
    #normcorr= 1
    crossdustsync = np.repeat(p[2*Nell:3*Nell],Ncross)*normcorr*(func.mbb_uK(nu_i, p[ellim+1], p[ellim+2],nu0=nuref) * func.PL_uK(nu_j, p[ellim+3],nu0=nurefs) + func.PL_uK(nu_i, p[ellim+3],nu0=nurefs) * func.mbb_uK(nu_j, p[ellim+1], p[ellim+2],nu0=nuref))
    lognui = np.log(nu_i/nuref)
    lognuj = np.log(nu_j/nuref)
    dx0 = func.dmbb_bT(nuref,p[ellim+2])
    dxi = func.dmbb_bT(nu_i,p[ellim+2])
    dxj = func.dmbb_bT(nu_j,p[ellim+2])
    temp = ampl*(np.repeat(p[:Nell],Ncross)+ (lognui+lognuj) * p[ellim+5]*PL_ell(ell,p[ellim+12])+ lognui*lognuj * p[ellim+6]*PL_ell(ell,p[ellim+13]))
    temp2= ampl*((dxi+dxj-2*dx0)*p[ellim+7]*PL_ell(ell,p[ellim+14])+(lognuj*(dxi-dx0)+lognui*(dxj-dx0))*p[ellim+8]*PL_ell(ell,p[ellim+15])+(dxi-dx0)*(dxj-dx0)*p[ellim+9]*PL_ell(ell,p[ellim+16]))
    crossdustsync2 = p[ellim+10]*PL_ell(ell,p[ellim+17])*(func.mbb_uK(nu_i,p[ellim+1],p[ellim+2],nu0=nuref)*lognui*func.PL_uK(nu_j,p[ellim+3],nu0=nurefs)+ func.PL_uK(nu_i,p[ellim+3],nu0=nurefs)*func.mbb_uK(nu_j,p[ellim+1],p[ellim+2],nu0=nuref)*lognuj)
    crossdustsync3 = p[ellim+11]*PL_ell(ell,p[ellim+18])*(func.mbb_uK(nu_i,p[ellim+1],p[ellim+2],nu0=nuref)*(dxi-dx0)*func.PL_uK(nu_j,p[ellim+3],nu0=nurefs)+ func.PL_uK(nu_i,p[ellim+3],nu0=nurefs)*func.mbb_uK(nu_j,p[ellim+1],p[ellim+2],nu0=nuref)*(dxj-dx0))
    model = temp + temp2 + sync+ crossdustsync+ crossdustsync2+ crossdustsync3+ DL_lensbin + p[ellim+4]*DL_tens
    return model

def func_ds_o1bts_all_ell(p, x1=None, x2=None,nuref=353,nurefs=23.,ell=None,Nell=None,DL_lensbin=None, DL_tens=None):
    nu_i=x1
    nu_j=x2
    ellim=3*Nell-1
    Ncross=len(nu_i)/Nell
    ampl = func.mbb_uK(nu_i, p[ellim+1], p[ellim+2],nu0=nuref) * func.mbb_uK(nu_j, p[ellim+1], p[ellim+2],nu0=nuref)
    sync = np.repeat(p[Nell:2*Nell],Ncross)*func.PL_uK(nu_i, p[ellim+3],nu0=nurefs) * func.PL_uK(nu_j, p[ellim+3],nu0=nurefs)
    normcorr= np.repeat(np.sqrt(abs(p[:Nell]*p[Nell:2*Nell])),Ncross)
    #normcorr= 1
    crossdustsync = np.repeat(p[2*Nell:3*Nell],Ncross)*normcorr*(func.mbb_uK(nu_i, p[ellim+1], p[ellim+2],nu0=nuref) * func.PL_uK(nu_j, p[ellim+3],nu0=nurefs) + func.PL_uK(nu_i, p[ellim+3],nu0=nurefs) * func.mbb_uK(nu_j, p[ellim+1], p[ellim+2],nu0=nuref))
    lognui =  np.log(nu_i/nuref)
    lognuj =  np.log(nu_j/nuref)
    lognuis = np.log(nu_i/nurefs)
    lognujs = np.log(nu_j/nurefs)
    dx0 = func.dmbb_bT(nuref,p[ellim+2])
    dxi = func.dmbb_bT(nu_i,p[ellim+2])
    dxj = func.dmbb_bT(nu_j,p[ellim+2])
    temp = ampl*(np.repeat(p[:Nell],Ncross)+ (lognui+lognuj) * p[ellim+5]*PL_ell(ell,p[ellim+12])+ lognui*lognuj * p[ellim+6]*PL_ell(ell,p[ellim+13]))
    temp2= ampl*((dxi+dxj-2*dx0)*p[ellim+7]*PL_ell(ell,p[ellim+14])+(lognuj*(dxi-dx0)+lognui*(dxj-dx0))*p[ellim+8]*PL_ell(ell,p[ellim+15])+(dxi-dx0)*(dxj-dx0)*p[ellim+9]*PL_ell(ell,p[ellim+16]))
    crossdustsync2 = normcorr*p[ellim+10]*PL_ell(ell,p[ellim+17])*(func.mbb_uK(nu_i,p[ellim+1],p[ellim+2],nu0=nuref)*lognui*func.PL_uK(nu_j,p[ellim+3],nu0=nurefs)+ func.PL_uK(nu_i,p[ellim+3],nu0=nurefs)*func.mbb_uK(nu_j,p[ellim+1],p[ellim+2],nu0=nuref)*lognuj)
    crossdustsync3 = normcorr*p[ellim+11]*PL_ell(ell,p[ellim+18])*(func.mbb_uK(nu_i,p[ellim+1],p[ellim+2],nu0=nuref)*(dxi-dx0)*func.PL_uK(nu_j,p[ellim+3],nu0=nurefs)+ func.PL_uK(nu_i,p[ellim+3],nu0=nurefs)*func.mbb_uK(nu_j,p[ellim+1],p[ellim+2],nu0=nuref)*(dxj-dx0))
    crossdustsync4 = normcorr*p[ellim+12]*PL_ell(ell,p[ellim+19])*(func.mbb_uK(nu_i,p[ellim+1],p[ellim+2])*lognujs*func.PL_uK(nu_j,p[ellim+3])+ func.PL_uK(nu_i,p[ellim+3])*func.mbb_uK(nu_j,p[ellim+1],p[ellim+2])*lognuis)/(func.PL_uK(nurefs,p[4])*func.mbb_uK(nuref,p[ellim+1],p[ellim+2]))
    crossdustsync5 = normcorr*p[ellim+13]*PL_ell(ell,p[ellim+20])*(func.mbb_uK(nu_i,p[ellim+1],p[ellim+2])*lognui*func.PL_uK(nu_j,p[ellim+3])*lognujs+ func.PL_uK(nu_i,p[ellim+3])*lognuis*func.mbb_uK(nu_j,p[ellim+1],p[ellim+2])*lognuj)/(func.PL_uK(nurefs,p[ellim+3])*func.mbb_uK(nuref,p[ellim+1],p[ellim+2]))
    crossdustsync6 = normcorr*p[ellim+14]*PL_ell(ell,p[ellim+21])*(func.mbb_uK(nu_i,p[ellim+1],p[ellim+2])*(dxi-dx0)*func.PL_uK(nu_j,p[ellim+3])*lognujs+ func.PL_uK(nu_i,p[ellim+3])*lognuis*func.mbb_uK(nu_j,p[ellim+1],p[ellim+2])*(dxj-dx0))/(func.PL_uK(nurefs,p[ellim+3])*func.mbb_uK(nuref,p[1],p[ellim+3]))
    syncmom = sync*((lognuis+lognujs) * p[ellim+15]*PL_ell(ell,p[ellim+22])+ lognuis*lognujs * p[ellim+16]*PL_ell(ell,p[ellim+23]))
    model = temp + temp2 + syncmom + crossdustsync+ crossdustsync2+ crossdustsync3+crossdustsync4+crossdustsync5+crossdustsync6+ DL_lensbin + p[ellim+4]*DL_tens
    return model
