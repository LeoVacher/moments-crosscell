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


instr_name='litebird_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
freq= instr['frequencies']

dusttype=[1,10,12]
syncrotype=[1,5,7]
nside=64
scale=10
fsky=0.8
Nlbin = 10
lmax = nside*3-1

if fsky==1:
    mask = np.ones(hp.nside2npix(nside))
else:
    mask = hp.read_map("./masks/mask_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale))

b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin,is_Dell=True)
leff = b.get_effective_ells()

def getmom_downgr(mom):
    momarr=np.array([np.zeros(hp.nside2npix(512)),mom.real,mom.imag])
    momdg=sim.downgrade_map(momarr,nside_in=512,nside_out=nside)
    return momdg

def get_dl_bb_mom(map1,map2):
    return sim.compute_cross_simple(getmom_downgr(map1)[1:],getmom_downgr(map2)[1:],mask,b)[3]
    
def getmom(dusttype, syncrotype):
    sky = pysm3.Sky(nside=512, preset_strings=['d%s'%(dusttype),'s%s'%(syncrotype)])
    skyd = pysm3.Sky(nside=512, preset_strings=['d%s'%dusttype])
    skys = pysm3.Sky(nside=512, preset_strings=['s%s'%syncrotype])
    dust = skyd.components[0]
    sync= skys.components[0]
    betamap =dust.mbb_index.value
    tempmap=dust.mbb_temperature.value
    betasmap =sync.pl_index.value
    if dusttype==12:
        nu0d=dust.freq_ref.value
        Ampl = dust.layers
        Amplcpxd = Ampl[:,1]+1j*Ampl[:,2]
    else:
        nu0d=dust.freq_ref_P.value
    nu0s=sync.freq_ref_P.value
        
    skyrefd = skyd.get_emission(nu0d * u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(nu0d*u.GHz)).value
    skyrefs = skys.get_emission(nu0s * u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(nu0s*u.GHz)).value

    model= np.array([sky.get_emission(freq[f] * u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq[f]*u.GHz)).value for f in range(len(freq))])
    skyrefcpxd=skyrefd[1]+1j*skyrefd[2]
    skyrefcpxs=skyrefs[1]+1j*skyrefs[2]
    betabar= 1.54
    tempbar=20
    betasbar=-3

    if dusttype ==12:
        mom1b = np.sum(Amplcpxd*(betamap-betabar),axis=0)
        mom2b = np.sum(Amplcpxd*(betamap-betabar)**2,axis=0)
        pmetbar=1/20
        pmetmap=1/tempmap
        mom1pmet = np.sum(Amplcpxd*(pmetmap-pmetbar),axis=0)

    else: 
        mom1b = skyrefcpxd*(betamap-betabar)
        mom2b = skyrefcpxd*(betamap-betabar)**2
        pmetbar=1/20
        pmetmap=1/tempmap
        mom1pmet = skyrefcpxd*(pmetmap-pmetbar)
    
    mom1bs = skyrefcpxs*(betasmap-betasbar)

    Ad= get_dl_bb_mom(skyrefcpxd,skyrefcpxd)
    As= get_dl_bb_mom(skyrefcpxs,skyrefcpxs)
    Asd=get_dl_bb_mom(skyrefcpxd,skyrefcpxs)/np.sqrt(Ad*As)
    w1bw1b = get_dl_bb_mom(mom1b,mom1b)
    Aw1b= get_dl_bb_mom(skyrefcpxd,mom1b)
    Asw1b= get_dl_bb_mom(skyrefcpxs,mom1b)

    Aw1p=get_dl_bb_mom(skyrefcpxd,mom1pmet)
    w1bw1p=get_dl_bb_mom(mom1b,mom1pmet)
    w1pw1p=get_dl_bb_mom(mom1pmet,mom1pmet)
    Asw1p= get_dl_bb_mom(skyrefcpxs,mom1pmet)

    return [Ad,As,Asd,w1bw1b,Aw1b,Aw1p,w1bw1p,w1pw1p,Asw1b,Asw1p]


analytical_mom=np.array([getmom(dusttype[d], syncrotype[d]) for d in range(len(dusttype))])

name=['A','A_s','A_sd','w1bw1b','Aw1b','Aw1t','w1bw1t','w1tw1t','Asw1b','Asw1t']

for d in range(len(dusttype)):
    arr= analytical_mom[d]
    momdict = dict(zip(name, arr))
    np.save('./analytical_mom/analytical_mom_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%s.npy'%(nside,fsky,scale,Nlbin,dusttype[d],syncrotype[d]),momdict)