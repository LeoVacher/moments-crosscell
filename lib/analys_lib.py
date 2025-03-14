import numpy as np
from mpfit import mpfit
import fitlib as ftl
import scipy
import matplotlib.pyplot as plt 
import basicfunc as func
from tqdm import tqdm
import mpi4py
from mpi4py import MPI
import plotlib as plib
import pymaster as nmt 
import pathlib

#contains all function for moment fitting.

#GENERAL #######################################################################################################################

def adaptafix(arr):
    """
    Return 1 if the moments are not detected and 0 if the moments are detected.
    Is used to chose if the moments should be fixed or fitted in the adaptative framework.
    :param arr: array or list of size N (number of sims) containing all the best-fit values for a moment coefficient.
    """
    med= np.median(arr)
    mad = scipy.stats.median_abs_deviation(arr)
    a= med+mad/2
    b= med-mad/2
    x=(a < 0 < b) or (b < 0 < a)
    if x==True:
        return 1
    else: 
        return 0

# FIT FUNCTIONS ##################################################################################################################

def fit_mom(kw,nucross,DL,Linv,p0,quiet=True,parallel=False,nside = 64, Nlbin = 10,fix=1,all_ell=False,adaptative=False,kwsave=""):
    """
    Fit using a first order moment expansion in both beta and T on a DL
    :param: kw, should be a string of the form 'X_Y' where X={d,s,ds} for dust,syncrotron or dust and syncrotron, and Y={o0,o1bt,o1bts} for order 0, first order in beta and T or first order in beta, T, betas
    :param: nucross, array of the cross-frequencies
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param Linv: Cholesky of the inverse covariance matrix
    :param quiet: display output of the fit for debugging
    :param: parallel, if true use mpi to parallelise the computation on number of simulations.
    :param nside: nside of the simulations
    :param Nlbin: binning of the spectra
    :param fix: fix or fit the spectral parameters. fix=0 fit them, fix=1 keep them fixed to the values contained in p0.
    :param all_ell: fit each multipole independently (False) or perform a single (longer) fit over all the multipole range (True).
    :param adaptive: if True use the results of a previous run to fit only the detected moments. 
    :param kwsave: keyword to save the results in the folder "best_fits".
    :return results: dictionnary containing A, beta, temp, Aw1b, w1bw1b, r and X2red for each (ell,n)
    """
    N,_,Nell=DL.shape
    nparam = len(p0)

    #ell array
    b = nmt.bins.NmtBin(nside=nside,lmax=nside*3-1,nlb=Nlbin)
    l = b.get_effective_ells()
    l=l[:Nell]
    #update keyword for load and save:
    kwf=kw+'_fix%s'%fix
    if all_ell==True:
        kwf=kwf+"_all_ell"
    if adaptative==True:
        kwf=kwf+'_adaptive'

    #create folder for parallel    
    if parallel==True:
        pathlib.Path('./best_fits/results_%s_%s.npy'%(kwsave,kwf)).mkdir(parents=True, exist_ok=True)

    # get cmb spectra:
    DL_lensbin, DL_tens= ftl.getDL_cmb(nside=nside,Nlbin=Nlbin)

    #get frequencies:
    ncross=len(nucross)
    nnus = int((-1 + np.sqrt(ncross * 8 + 1)) / 2.)
    posauto = [int(nnus * i - i * (i + 1) / 2 + i) for i in range(nnus)]
    nu = nucross[posauto]
    freq_pairs = np.array([(i, j) for i in range(nnus) for j in range(i, nnus)])
    nu_i = nu[freq_pairs[:, 0]]
    nu_j = nu[freq_pairs[:, 1]]

    #select function to fit:
    funcfit= eval('ftl.func_'+kw)
     
    if all_ell==True:
        #put arrays in NcrossxNell shape for all-ell fit
        nu_i = np.tile(nu_i, Nell)
        nu_j = np.tile(nu_j, Nell)
        DL_lensbin= np.repeat(DL_lensbin[:Nell],ncross)
        DL_tens= np.repeat(DL_tens[:Nell],ncross)
        DLdcflat = np.zeros([N,Nell*ncross])
        for i in range(N):
            DLdcflat[i] = np.swapaxes(DL[i,:,:],0,1).flatten()

    if all_ell==False:
        #initialize parameters and chi2:
        paramiterl=np.zeros((Nell,N,nparam))
        chi2l=np.zeros((Nell,N))

        #set initial values:   
        parinfopl =  [{'value':p0[i], 'fixed':0} for i in range(nparam)] #fg params
        parinfopl[0]= {'value':p0[0], 'fixed':0,'limited':[1,0],'limits':[0,np.inf]} #Ad
        parinfopl[1]= {'value':p0[1], 'fixed':fix,'limited':[1,1],'limits':[0.5,2]} #betad
        parinfopl[2]= {'value':1/p0[2], 'fixed':fix,'limited':[1,1],'limits':[1/100,1/3]} #1/Td
        parinfopl[3]= {'value':p0[3], 'fixed':0,'limited':[1,0],'limits':[0,np.inf]} #As
        parinfopl[4]= {'value':p0[4], 'fixed':fix,'limited':[1,1],'limits':[-5,-2]} #betas    
        parinfopl = np.array([parinfopl for i in range(Nell)])
        if adaptative==True:
            res0=np.load('./best_fits/results_%s_%s.npy'%(kwsave,kwf),allow_pickle=True).item()
            keys= res0.keys()
            for k in range(6,len(res0.keys())-2):
                for L in range(Nell):
                    fixmom=adaptafix(res0[list(keys)[k]][L])
                    parinfopl[L][k]= {'value':0, 'fixed':fixmom}
        #for parallel:
        if parallel==True:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            perrank = math.ceil(N/size)
            Nmin= rank*perrank
            Nmax= (rank+1)*perrank
        else:
            Nmin=0
            Nmax=N
    
        #perform the fit

        for n in tqdm(range(Nmin,Nmax)):
            for L in range(Nell):
                # first o1 fit, dust fixed, mom free, r fixed
                fa = {'x1':nu_i, 'x2':nu_j, 'y':DL[n,:,L], 'err': Linv[L],'ell':L, 'DL_lensbin': DL_lensbin, 'DL_tens': DL_tens,'model_func':funcfit}
                m = mpfit(ftl.lkl_mpfit,parinfo= list(parinfopl[L]) ,functkw=fa,quiet=quiet)
                paramiterl[L,n]= m.params
                chi2l[L,n]=m.fnorm/m.dof            
        
        #return result dictionnary:

        if kw=='ds_o0':
            results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : 1/paramiterl[:,:,2], 'A_s': paramiterl[:,:,3], 'beta_s': paramiterl[:,:,4],'A_sd' : paramiterl[:,:,5], 'r':paramiterl[:,:,6], 'X2red': chi2l}
        elif kw=='ds_o1bt':
            results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : 1/paramiterl[:,:,2], 'A_s':paramiterl[:,:,3] , 'beta_s':paramiterl[:,:,4], 'A_sd':paramiterl[:,:,5], 'Aw1b' : paramiterl[:,:,6], 'w1bw1b' : paramiterl[:,:,7],'Aw1t' : paramiterl[:,:,8],'w1bw1t' : paramiterl[:,:,9],'w1tw1t' : paramiterl[:,:,10],'Asw1b' : paramiterl[:,:,11],'Asw1t' : paramiterl[:,:,12],'r' : paramiterl[:,:,13], 'X2red': chi2l}
        elif kw=='ds_o1bts':
            results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : 1/paramiterl[:,:,2], 'A_s':paramiterl[:,:,3] , 'beta_s':paramiterl[:,:,4], 'A_sd':paramiterl[:,:,5], 'Aw1b' : paramiterl[:,:,6], 'w1bw1b' : paramiterl[:,:,7],'Aw1t' : paramiterl[:,:,8],'w1bw1t' : paramiterl[:,:,9],'w1tw1t' : paramiterl[:,:,10],'Asw1bs' : paramiterl[:,:,11],'w1bsw1bs' : paramiterl[:,:,12],'Asw1b' : paramiterl[:,:,13],'Asw1t' : paramiterl[:,:,14],'Adw1s' : paramiterl[:,:,15],'w1bw1s' : paramiterl[:,:,16],'w1sw1T' : paramiterl[:,:,17],'r' : paramiterl[:,:,18], 'X2red': chi2l}
        else:
            print('unexisting keyword')

    if all_ell==True:
        funcfit= eval('ftl.func_'+kw+'_all_ell')

        #set initial values:
        parinfopl =  []
        [parinfopl.append({'value':p0[0], 'fixed':0,'limited':[1,0],'limits':[0,np.inf]}) for i in range(Nell)] #A_d
        [parinfopl.append({'value':p0[3], 'fixed':0,'limited':[1,0],'limits':[0,np.inf]}) for i in range(Nell)] #A_s
        [parinfopl.append({'value':p0[5], 'fixed':0,'limited':[0,0],'limits':[-np.inf,np.inf]}) for i in range(Nell)] #A_sd
        parinfopl.append({'value':p0[1], 'fixed':fix,'limited':[1,1],'limits':[0.5,2]}) #betad
        parinfopl.append({'value':1/p0[2], 'fixed':fix,'limited':[1,1],'limits':[1/100,3]}) #1/Td
        parinfopl.append({'value':p0[4], 'fixed':fix,'limited':[1,1],'limits':[-5,-2]}) #betas    
        parinfopl.append({'value':p0[5], 'fixed':0}) #r 
        if kw=='ds_o1bt':
            [parinfopl.append({'value':0,'fixed':0}) for i in range(7)] #moments and power-law indices 
            [parinfopl.append({'value':-0.5,'fixed':0,'limited':[1,1],'limits':[-4,0.1]}) for i in range(7)] #power-law indices 
        elif kw=='ds_o1bts':
            [parinfopl.append({'value':0,'fixed':0}) for i in range(10)] #moments and power-law indices 
            [parinfopl.append({'value':0,'fixed':0}) for i in range(10)] #power-law indices 
        chi2=np.zeros(N)
        paramiter=np.zeros((N,len(parinfopl)))
        
        #for parallel:
        if parallel==True:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            perrank = math.ceil(N/size)
            Nmin= rank*perrank
            Nmax= (rank+1)*perrank
        else:
            Nmin=0
            Nmax=N
        
        #perform the fit

        for n in tqdm(range(Nmin,Nmax)):
            # first o1 fit, dust fixed, mom free, r fixed
            fa = {'x1':nu_i, 'x2':nu_j, 'y':DLdcflat[n], 'err': Linv, 'DL_lensbin': DL_lensbin, 'DL_tens': DL_tens,'ell':np.repeat(l,ncross),'Nell':Nell,'model_func': funcfit}
            m = mpfit(ftl.lkl_mpfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiter[n]= m.params
            chi2[n]=m.fnorm/m.dof            
        
        #return result dictionnary:

        if kw=='ds_o0':
            results={'A' : np.swapaxes(paramiter[:,:Nell],0,1), 'beta' : paramiter[:,3*Nell], 'temp' : 1/paramiter[:,3*Nell+1], 'A_s': np.swapaxes(paramiter[:,Nell:2*Nell],0,1), 'beta_s': paramiter[:,3*Nell+2],'A_sd' : np.swapaxes(paramiter[:,2*Nell:3*Nell],0,1), 'r':paramiter[:,3*Nell+3], 'X2red': chi2}
        elif kw=='ds_o1bt':
            results_o0=        {'A' : np.swapaxes(paramiter[:,:Nell],0,1), 'beta' : paramiter[:,3*Nell], 'temp' : 1/paramiter[:,3*Nell+1], 'A_s':np.swapaxes(paramiter[:,Nell:2*Nell],0,1) , 'beta_s':paramiter[:,3*Nell+2], 'A_sd':np.swapaxes(paramiter[:,2*Nell:3*Nell],0,1)}
            results_mom =   {'Aw1b' : paramiter[:,3*Nell+4], 'w1bw1b' : paramiter[:,3*Nell+5],'Aw1t' : paramiter[:,3*Nell+6],'w1bw1t' : paramiter[:,3*Nell+7],'w1tw1t' : paramiter[:,3*Nell+8],'Asw1b' : paramiter[:,3*Nell+9],'Asw1t' : paramiter[:,3*Nell+10]}
            results_mom_pl= {'alpha_Aw1b' : paramiter[:,3*Nell+11], 'alpha_w1bw1b' : paramiter[:,3*Nell+12],'alpha_Aw1t' : paramiter[:,3*Nell+13],'alpha_w1bw1t' : paramiter[:,3*Nell+14],'alpha_w1tw1t' : paramiter[:,3*Nell+15],'alpha_Asw1b' : paramiter[:,3*Nell+16],'alpha_Asw1t' : paramiter[:,3*Nell+17],'r' : paramiter[:,3*Nell+3], 'X2red': chi2}
            results = {**results_o0,**results_mom,**results_mom_pl}
        elif kw=='ds_o1bts':
            results_o0 =     {'A' : np.swapaxes(paramiter[:,:Nell],0,1), 'beta' : paramiter[:,3*Nell], 'temp' : 1/paramiter[:,3*Nell+1], 'A_s':np.swapaxes(paramiter[:,Nell:2*Nell],0,1) , 'beta_s':paramiter[:,3*Nell+2], 'A_sd':np.swapaxes(paramiter[:,2*Nell:3*Nell],0,1)}
            results_mom=     {'Aw1b' : paramiter[:,3*Nell+4], 'w1bw1b' : paramiter[:,3*Nell+4],'Aw1t' : paramiter[:,3*Nell+6],'w1bw1t' : paramiter[:,3*Nell+7],'w1tw1t' : paramiter[:,3*Nell+8],'Asw1bs' : paramiter[:,3*Nell+9],'w1bsw1bs' : paramiter[:,3*Nell+10],'Asw1b' : paramiter[:,3*Nell+11],'Asw1t' : paramiter[:,3*Nell+12],'Adw1s' : paramiter[:,3*Nell+13],'w1bw1s' : paramiter[:,3*Nell+14],'w1sw1T' : paramiter[:,3*Nell+15],'r' : paramiter[:,3*Nell+3], 'X2red': chi2}
            results_mom_pl = {}
            results = {**results_o0,**results_mom,**results_mom_pl}
        else:
            raise ValueError('unexisting keyword')

    #save and plot results
    
    if parallel==True:
        np.save('best_fits/results_%s_%s_p0/res%s.npy'%(kwsave,kwf,rank))    
    else:
        np.save('./best_fits/results_%s_%s.npy'%(kwsave,kwf),results)
        plib.plotrespdf(l[:Nell],[results],['%s-%s'%(kwsave,kwf)],['darkorange'])
        if all_ell==True:
            plib.plotr_hist(results,color='darkorange',save=True,kwsave='%s%s'%(kwsave,kwf))
        else:
            plib.plotr_gaussproduct(results,Nmax=Nell,debug=False,color='darkorange',save=True,kwsave='%s-%s'%(kwsave,kwf))
    return results
