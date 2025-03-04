import healpy as hp
import numpy as np
import pymaster as nmt

def downgrade_alm(input_alm,nside_in,nside_out):
    """
    This is a Function to downgrade Alm correctly.
    nside_in must be bigger than nside_out.
    In this function, lmax_in = 3*nside_in-1 , lmax_out = 3*nside_out-1 .
    input_alm must be lmax = lmax_in and output_alm must be lmax = lmax_out.
    This function get only values in the range 0 < l < lmax_out from input_alm,
    and put these values into output_alm which has range 0 < l < lmax_out.
    """
    lmax_in = nside_in*3-1
    lmax_out = nside_out*3-1
    output_alm = np.zeros((3,hp.sphtfunc.Alm.getsize(lmax_out)),dtype=complex)
    
    for m in range(lmax_out+1):
        idx_1_in = hp.sphtfunc.Alm.getidx(lmax_in,m ,m)
        idx_2_in = hp.sphtfunc.Alm.getidx(lmax_in,lmax_out ,m)

        idx_1_out = hp.sphtfunc.Alm.getidx(lmax_out,m ,m)
        idx_2_out = hp.sphtfunc.Alm.getidx(lmax_out,lmax_out ,m)

        output_alm[:,idx_1_out:idx_2_out+1] = input_alm[:,idx_1_in:idx_2_in+1]
    return output_alm

def downgrade_map(input_map,nside_out,nside_in):
    """
    This is a Function to downgrade map correctly in harmonic space.
    nside_in must be bigger than nside_out.
    input_map must have nside_in.
    output_map has nside_out as Nside
    """
    #  nside_in= hp.npix2nside(len(input_map))
    input_alm = hp.map2alm(input_map)  #input map → input alm
    output_alm = downgrade_alm(input_alm,nside_in,nside_out) # input alm → output alm (decrease nside)
    output_map = hp.alm2map(output_alm,nside=nside_out)#  output alm → output map
    return output_map

def compute_master(f_a, f_b, wsp):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled
   

Pr = "/global/homes/l/leovchr/"
Pathsave=Pr+ '/codes/moments-crosscell/CLsimus/'

load=True
nside = 64
Npix = hp.nside2npix(nside)
N=249
lmax = 3*nside-1
scale = 10
Nlbin = 10
fsky = 0.8
complexity='medium_complexity'   #should be 'baseline', 'high_complexity' or 'medium_complexity'
kw=''
r=0

folder= "/global/cfs/cdirs/litebird/simulations/maps/E_modes_postptep/2ndRelease/mock_splits_coadd_sims/e2e_noise/%s"%complexity
b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
leff = b.get_effective_ells()

bands = ['LFT_L1-040', 'LFT_L2-050', 'LFT_L1-060', 'LFT_L3-068','LFT_L2-068', 'LFT_L4-078', 'LFT_L1-078', 'LFT_L3-089', 'LFT_L2-089','LFT_L4-100', 'LFT_L3-119', 'LFT_L4-140',
         'MFT_M1-100', 'MFT_M2-119', 'MFT_M1-140', 'MFT_M2-166', 'MFT_M1-195',
         'HFT_H1-195', 'HFT_H2-235', 'HFT_H1-280', 'HFT_H2-337', 'HFT_H3-402']

freq=np.array([40,50,60,58,68,78,78,89,89,100,119,140,100,119,140,166,195,195,235,280,337,403])
N_freqs=len(bands)

mask = hp.read_map("./masks/mask_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale))

maptot= np.zeros((N_freqs,3,Npix))

for i in range(N_freqs):
    maptoti= hp.read_map(folder+"/0000/"+"coadd_maps_LB_%s_cmb_e2e_sims_fg_%s_wn_1f_binned_030mHz_0000_full.fits"%(bands[i],complexity),field=(0,1,2))
    maptot[i]= downgrade_map(maptoti,nside_in=512,nside_out=nside)
maptot=maptot[:,1:]

wsp = nmt.NmtWorkspace()
wsp.compute_coupling_matrix(nmt.NmtField(mask, 1*mapfg[0],purify_e=False, purify_b=True), nmt.NmtField(mask,1*mapfg[0],purify_e=False, purify_b=True), b)

Ncross=int(N_freqs*(N_freqs+1)/2)

if load ==True:
    CLdc= 2*np.pi*np.load(Pathsave+'DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy'%(nside,fsky,scale,Nlbin,complexity[0],complexity[0]))/leff/(leff+1)  
    kini=np.argwhere(CLdc == 0)[0,0]
else:
    kini=0
    CLdc=np.zeros((N,Ncross,len(leff)))
    
maptotfull= np.zeros((N_freqs,3,Npix))
maptot_HM1= np.zeros((N_freqs,3,Npix))
maptot_HM2= np.zeros((N_freqs,3,Npix))

beam = np.array([70.5,58.5,51.1,41.6,47.1,36.9, 43.8, 33.0 , 41.5, 30.2,26.3, 23.7, 37.8,33.6,30.8,28.9,28.0,28.6,24.7,22.5,20.9,17.9])

BL=[]
for i in range(len(beam)):
    bl=np.array(hp.gauss_beam(beam[i]*np.pi/(60*180),lmax = lmax,pol=True))
    bl= b.bin_cell(bl[:,1])[0:len(leff)]
    BL.append(bl)
BL = np.array(BL)

for k in range(kini,N):
    a='000'+"%s"%k
    if len(a)>4:
        a=a.replace("000","00")
    if k>=100:
        a='0'+str(k)
    print(a)
    
    #gérer list et concatenate

    for i in range(N_freqs):
        maptotf= hp.read_map(folder+"/%s/"%a+"coadd_maps_LB_%s_cmb_e2e_sims_fg_%s_wn_1f_binned_030mHz_%s_full.fits"%(bands[i],complexity,a),field=(0,1,2))
        maptot_f1= hp.read_map(folder+"/%s/"%a+"coadd_maps_LB_%s_cmb_e2e_sims_fg_%s_wn_1f_binned_030mHz_%s_splitA.fits"%(bands[i],complexity,a),field=(0,1,2))
        maptot_f2= hp.read_map(folder+"/%s/"%a+"coadd_maps_LB_%s_cmb_e2e_sims_fg_%s_wn_1f_binned_030mHz_%s_splitB.fits"%(bands[i],complexity,a),field=(0,1,2))

        maptotfull[i]= downgrade_map(maptotf,nside_in=512,nside_out=nside)
        maptot_HM1[i]= downgrade_map(maptot_f1,nside_in=512,nside_out=nside)
        maptot_HM2[i]= downgrade_map(maptot_f2,nside_in=512,nside_out=nside)

    z=0
    for i in range(0,N_freqs):
        for j in range(i,N_freqs):
            #to be update with simlib functions
            if i!=j:
                CLdc[k,z] = np.array((compute_master(nmt.NmtField(mask, 1*maptotfull[i,1:],purify_e=False, purify_b=True), nmt.NmtField(mask,1*maptotfull[j,1:],purify_e=False, purify_b=True), wsp))[3])
                CLdc[k,z] = CLdc[k,z]/BL[i]/BL[j]
            if i==j:
                CLdc[k,z] = np.array((compute_master(nmt.NmtField(mask, 1*maptot_HM1[i,1:],purify_e=False, purify_b=True), nmt.NmtField(mask,1*maptot_HM2[j,1:],purify_e=False, purify_b=True), wsp))[3])
                CLdc[k,z] = CLdc[k,z]/BL[i]/BL[j]                
            z = z +1
    np.save(Pathsave+'DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy'%(nside,fsky,scale,Nlbin,complexity[0],complexity[0]),leff*(leff+1)*CLdc/2/np.pi) 