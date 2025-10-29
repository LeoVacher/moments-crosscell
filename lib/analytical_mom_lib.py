#Import
import sys
sys.path.append("./lib")
import basicfunc as func
import plotlib as plib
import numpy as np
import pylab
pylab.rcParams['figure.figsize'] = 12, 16
import pymaster as nmt
import healpy as hp
import pysm3
import pysm3.units as u
import simu_lib as sim
import scipy.stats as st
import simu_lib as sim
import basicfunc as func


def getmom_downgr(mom, nside, nside_pysm):
    momarr = np.array([np.zeros(hp.nside2npix(nside_pysm)), mom.real, mom.imag])
    momdg = sim.downgrade_map(momarr, nside_out=nside, nside_in=nside_pysm)
    return momdg

def get_dl_mom(mom1,mom2,nside,mask,b, mode='BB'):
    sp_dict = {'EE':[0,True,False], 'EB':[1,True,True], 'BE': [2,True,True], 'BB': [3,False,True]}
    sp, purify_e, purify_b = sp_dict.get(mode, None)
    return sim.compute_cross_simple(mom1[1:], mom2[1:], mask, b, purify_e=purify_e, purify_b=purify_b)[sp]
  
def getmom(dusttype, synctype, betabar, tempbar, betasbar, mask, Nlbin=10,nside=64,nu0d=353.,nu0s=23.,momsync=True, mode='BB'):
   # to do : add beam effect and add EE moments if needed
    lmax = nside*3-1
    b = nmt.NmtBin.from_lmax_linear(lmax=lmax,nlb=Nlbin,is_Dell=True)
    nside_pysm = 512
    if dusttype >= 9 or synctype >= 4:
        nside_pysm = 2048
    skyd = pysm3.Sky(nside=nside_pysm, preset_strings=['d%s'%dusttype])
    skys = pysm3.Sky(nside=nside_pysm, preset_strings=['s%s'%synctype])
    dust = skyd.components[0]
    sync = skys.components[0]
    betamap = dust.mbb_index.value
    tempmap = dust.mbb_temperature.value
    betasmap = sync.pl_index.value
    pmetbar = 1/tempbar
    pmetmap = 1/tempmap
    if dusttype==12:
        Ampl = dust.layers.value * func.unit_conversion(353, input_unit='uK_RJ', output_unit='uK_CMB')
        Amplcpxd = (Ampl[:,1]+1j*Ampl[:,2]) / func.mbb_uK(353, betamap, pmetmap, nu0=nu0d)
    skyrefd = skyd.get_emission(nu0d * u.GHz).value * func.unit_conversion(nu0d, input_unit='uK_RJ', output_unit='uK_CMB')
    skyrefs = skys.get_emission(nu0s * u.GHz).value * func.unit_conversion(nu0s, input_unit='uK_RJ', output_unit='uK_CMB')

    skyrefcpxd = skyrefd[1]+1j*skyrefd[2]
    skyrefcpxs = skyrefs[1]+1j*skyrefs[2]

    if dusttype == 12:
        mom1b = np.sum(Amplcpxd*(betamap-betabar),axis=0)
        mom2b = np.sum(Amplcpxd*(betamap-betabar)**2,axis=0)
        mom1pmet = np.sum(Amplcpxd*(pmetmap-pmetbar),axis=0)

    else: 
        mom1b = skyrefcpxd*(betamap-betabar)
        mom2b = skyrefcpxd*(betamap-betabar)**2
        mom1pmet = skyrefcpxd*(pmetmap-pmetbar)
    
    mom1bs = skyrefcpxs*(betasmap-betasbar)

    #amplitudes:
    skyrefcpxd = getmom_downgr(skyrefcpxd, nside, nside_pysm)
    skyrefcpxs = getmom_downgr(skyrefcpxs, nside, nside_pysm)
    Ad = get_dl_mom(skyrefcpxd,skyrefcpxd,nside,mask,b, mode=mode)
    As = get_dl_mom(skyrefcpxs,skyrefcpxs,nside,mask,b, mode=mode)
    Asd = get_dl_mom(skyrefcpxd,skyrefcpxs,nside,mask,b, mode=mode)/np.sqrt(Ad*As)
    
    #dust beta moments:
    mom1b = getmom_downgr(mom1b, nside, nside_pysm)
    w1bw1b = get_dl_mom(mom1b,mom1b,nside,mask,b, mode=mode)
    Aw1b = get_dl_mom(skyrefcpxd,mom1b,nside,mask,b, mode=mode)
    Asw1b = get_dl_mom(skyrefcpxs,mom1b,nside,mask,b, mode=mode)

    #dust 1/temp moments:
    mom1pmet = getmom_downgr(mom1pmet, nside, nside_pysm)
    Aw1p = get_dl_mom(skyrefcpxd,mom1pmet,nside,mask,b, mode=mode)
    w1bw1p = get_dl_mom(mom1b,mom1pmet,nside,mask,b, mode=mode)
    w1pw1p = get_dl_mom(mom1pmet,mom1pmet,nside,mask,b, mode=mode)
    Asw1p = get_dl_mom(skyrefcpxs,mom1pmet,nside,mask,b, mode=mode)

    #dust spectral parameters:
    beta_d = betabar + Aw1b / Ad
    T_d = 1 / (pmetbar + Aw1p / Ad)

    #syncrotron beta moments:

    if momsync:
        mom1bs = getmom_downgr(mom1bs, nside, nside_pysm)
        Aw1bs = get_dl_mom(skyrefcpxd,mom1bs,nside,mask,b, mode=mode)    
        Asw1bs = get_dl_mom(skyrefcpxs,mom1bs,nside,mask,b, mode=mode)    
        w1bw1bs = get_dl_mom(mom1b,mom1bs,nside,mask,b, mode=mode)
        w1bsw1bs = get_dl_mom(mom1bs,mom1bs,nside,mask,b, mode=mode)
        Asw1bs = get_dl_mom(skyrefcpxs,mom1bs,nside,mask,b, mode=mode)  
        w1pw1bs = get_dl_mom(mom1pmet,mom1bs,nside,mask,b, mode=mode)
        beta_s = betasbar + Asw1bs / As
        analytical_mom = np.array([Ad,beta_d,T_d,As,beta_s,Asd,w1bw1b,Aw1b,Aw1p,w1bw1p,w1pw1p,Asw1b,Asw1p,Asw1bs,w1bsw1bs,Aw1bs,Asw1bs,w1pw1bs,w1bw1bs])
        name = ['A_d','beta_d','T_d','A_s','beta_s','A_sd','w1bw1b','Aw1b','Aw1t','w1bw1t','w1tw1t','Asw1b','Asw1t','Asw1bs','w1bsw1bs','Aw1bs','Asw1bs','w1tw1bs','w1bw1bs']

    else:
        analytical_mom = np.array([Ad,beta_d,T_d,As,Asd,w1bw1b,Aw1b,Aw1p,w1bw1p,w1pw1p,Asw1b,Asw1p])
        name = ['A_d','beta_d','T_d','A_s','A_sd','w1bw1b','Aw1b','Aw1t','w1bw1t','w1tw1t','Asw1b','Asw1t']

    momdict = dict(zip(name, analytical_mom))
    return momdict