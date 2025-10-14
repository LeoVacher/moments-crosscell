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


def getmom_downgr(mom,nside):
    momarr=np.array([np.zeros(hp.nside2npix(512)),mom.real,mom.imag])
    momdg=sim.downgrade_map(momarr,nside_in=512,nside_out=nside)
    return momdg

def get_dl_bb_mom(map1,map2,nside,mask,b):
    return sim.compute_cross_simple(getmom_downgr(map1,nside)[1:],getmom_downgr(map2,nside)[1:],mask,b)[3]
  
def getmom(dusttype, syncrotype, betabar, tempbar, betasbar, mask, Nlbin=10,nside=64,nu0d=353.,nu0s=23.,momsync=True):
   # to do : add beam effect and add EE moments if needed
    lmax = nside*3-1
    b = nmt.NmtBin.from_lmax_linear(lmax=lmax,nlb=Nlbin,is_Dell=True)
    sky = pysm3.Sky(nside=512, preset_strings=['d%s'%(dusttype),'s%s'%(syncrotype)])
    skyd = pysm3.Sky(nside=512, preset_strings=['d%s'%dusttype])
    skys = pysm3.Sky(nside=512, preset_strings=['s%s'%syncrotype])
    dust = skyd.components[0]
    sync = skys.components[0]
    betamap = dust.mbb_index.value
    tempmap = dust.mbb_temperature.value
    betasmap = sync.pl_index.value
    if dusttype==12:
        Ampl = dust.layers
        Amplcpxd = Ampl[:,1]+1j*Ampl[:,2]        
    skyrefd = skyd.get_emission(nu0d * u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(nu0d*u.GHz)).value
    skyrefs = skys.get_emission(nu0s * u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(nu0s*u.GHz)).value

    skyrefcpxd = skyrefd[1]+1j*skyrefd[2]
    skyrefcpxs = skyrefs[1]+1j*skyrefs[2]

    if dusttype == 12:
        mom1b = np.sum(Amplcpxd*(betamap-betabar),axis=0)
        mom2b = np.sum(Amplcpxd*(betamap-betabar)**2,axis=0)
        pmetbar = 1/tempbar
        pmetmap = 1/tempmap
        mom1pmet = np.sum(Amplcpxd*(pmetmap-pmetbar),axis=0)

    else: 
        mom1b = skyrefcpxd*(betamap-betabar)
        mom2b = skyrefcpxd*(betamap-betabar)**2
        pmetbar = 1/tempbar
        pmetmap = 1/tempmap
        mom1pmet = skyrefcpxd*(pmetmap-pmetbar)
    
    mom1bs = skyrefcpxs*(betasmap-betasbar)

    #amplitudes:
    Ad = get_dl_bb_mom(skyrefcpxd,skyrefcpxd,nside,mask,b)
    As = get_dl_bb_mom(skyrefcpxs,skyrefcpxs,nside,mask,b)
    Asd = get_dl_bb_mom(skyrefcpxd,skyrefcpxs,nside,mask,b)/np.sqrt(Ad*As)
    
    #dust beta moments:
    w1bw1b = get_dl_bb_mom(mom1b,mom1b,nside,mask,b)
    Aw1b = get_dl_bb_mom(skyrefcpxd,mom1b,nside,mask,b)
    Asw1b = get_dl_bb_mom(skyrefcpxs,mom1b,nside,mask,b)

    #dust 1/temp moments:    
    Aw1p = get_dl_bb_mom(skyrefcpxd,mom1pmet,nside,mask,b)
    w1bw1p = get_dl_bb_mom(mom1b,mom1pmet,nside,mask,b)
    w1pw1p = get_dl_bb_mom(mom1pmet,mom1pmet,nside,mask,b)
    Asw1p = get_dl_bb_mom(skyrefcpxs,mom1pmet,nside,mask,b)

    #syncrotron beta moments:

    if momsync:
        Aw1bs = get_dl_bb_mom(skyrefcpxd,mom1bs,nside,mask,b)    
        Asw1bs = get_dl_bb_mom(skyrefcpxs,mom1bs,nside,mask,b)    
        w1bw1bs = get_dl_bb_mom(mom1b,mom1bs,nside,mask,b)
        w1bsw1bs = get_dl_bb_mom(mom1bs,mom1bs,nside,mask,b)
        Asw1bs = get_dl_bb_mom(skyrefcpxs,mom1bs,nside,mask,b)  
        w1bsw1p = get_dl_bb_mom(mom1bs,mom1pmet,nside,mask,b)
        analytical_mom = np.array([Ad,As,Asd,w1bw1b,Aw1b,Aw1p,w1bw1p,w1pw1p,Asw1b,Asw1p,Asw1bs,w1bsw1bs,Aw1bs,Asw1bs,w1bsw1p,w1bw1bs])
        name = ['A_d','A_s','A_sd','w1bw1b','Aw1b','Aw1t','w1bw1t','w1tw1t','Asw1b','Asw1t','Asw1bs','w1bsw1bs','Aw1bs','Asw1bs','w1bsw1t','w1bw1bs']

    else:
        analytical_mom = np.array([Ad,As,Asd,w1bw1b,Aw1b,Aw1p,w1bw1p,w1pw1p,Asw1b,Asw1p])
        name = ['A_d','A_s','A_sd','w1bw1b','Aw1b','Aw1t','w1bw1t','w1tw1t','Asw1b','Asw1t']

    momdict = dict(zip(name, analytical_mom))
    return momdict