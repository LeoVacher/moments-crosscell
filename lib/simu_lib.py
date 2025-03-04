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

def compute_master(f_a, f_b, wsp,coupled=False):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    if coupled==False:
        return cl_decoupled
    elif coupled==True:
        return cl_coupled

def get_wsp(map_FM1,map_FM2,map_HM1,map_HM2,mask,b):
    N_freqs=len(map_HM1)
    wsp=[]
    for i in range(0,N_freqs): 
        for j in range(i,N_freqs):
            w_temp = nmt.NmtWorkspace()
            if i != j :
                w_temp.compute_coupling_matrix(nmt.NmtField(mask, 1*map_FM1[i],purify_e=False, purify_b=True), nmt.NmtField(mask,1*map_FM2[j],purify_e=False, purify_b=True), b)
            if i==j :
                w_temp.compute_coupling_matrix(nmt.NmtField(mask, 1*map_HM1[i],purify_e=False, purify_b=True), nmt.NmtField(mask, 1*map_HM2[j],purify_e=False, purify_b=True), b)
            wsp.append(w_temp)
    wsp=np.array(wsp)

def computecross(map_FM1,map_FM2,map_HM1,map_HM2,wsp,fact_DL=1.,coupled=False,modes='BB'):
    N_freqs=len(map_HM1)
    Ncross=int(N_freqs*(N_freqs+1)/2)
    CLcross=np.zeros((Ncross,19))
    z=0
    sp_dict = {'EE': 0, 'EB': 1, 'BE':2, 'BB': 3}
    sp = sp_dict.get(mode, None)
    if sp!=None:
        for i in range(0,N_freqs):
            for j in range(i,N_freqs):
                if i != j :
                    CLcross[z]=np.array((sim.compute_master(nmt.NmtField(mask, 1*map_FM1[i],purify_e=False, purify_b=True), nmt.NmtField(mask, 1*map_FM2[j],purify_e=False, purify_b=True), wsp[z],coupled=coupled))[sp])
                if i==j :
                    CLcross[z]=np.array((sim.compute_master(nmt.NmtField(mask, 1*map_HM1[i],purify_e=False, purify_b=True), nmt.NmtField(mask, 1*map_HM2[j],purify_e=False, purify_b=True), wsp[z],coupled=coupled))[sp])
                z = z +1
    elif sp==None:
        for i in range(0,N_freqs):
            for j in range(i,N_freqs):
                if i != j :
                    CLcross[z]=np.array((sim.compute_master(nmt.NmtField(mask, 1*map_FM1[i],purify_e=False, purify_b=True), nmt.NmtField(mask, 1*map_FM2[j],purify_e=False, purify_b=True), wsp[z],coupled=coupled)))
                if i==j :
                    CLcross[z]=np.array((sim.compute_master(nmt.NmtField(mask, 1*map_HM1[i],purify_e=False, purify_b=True), nmt.NmtField(mask, 1*map_HM2[j],purify_e=False, purify_b=True), wsp[z],coupled=coupled)))
                z = z +1

    return fact_Dl*CLcross
