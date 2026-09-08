import numpy as np
from mpfit import mpfit
import fitlib as ftl
import scipy
import matplotlib.pyplot as plt 
import basicfunc as func
from tqdm import tqdm, trange
import mpi4py
from mpi4py import MPI
import plotlib as plib
import pymaster as nmt 
import pathlib
import analytical_mom_lib as anmomlib
import re
import healpy as hp
import covlib as cvl
import sympy as sym
from sympy.physics.wigner import wigner_3j
import astropy.constants as const

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
    if x:
        return 1
    else: 
        return 0

# FIT FUNCTIONS ##################################################################################################################

def fit_mom(kw,nucross,DL,Linv,p0,quiet=True,parallel=False,nside = 64, Nlbin = 10,fix=1,all_ell=False,adaptative=False,kwsave="",plotres=False,mompl=False,iterate=False,nu0d=353.,nu0s=23.,fixr=0,cmb_e2e=False,gnilc=False):
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
    :param adaptative: if True use the results of a previous run to fit only the detected moments. 
    :param kwsave: keyword to save the results in the folder "best_fits". Must be of the format dasb_fsky_kw, where a and b are the dusttype and synctype and fsky is the sky fraction between 0 and 1.
    :param plotres: if true, plot and save the results in pdf format.
    :param mompl: only for allell case, fit moments as power-laws in ell.
    :param iterate: if True, iterate the fit of the moments to estimate the best pivot value.
    :param fixr: if 1, fix the tensor to scalar ratio (r) to zero and does not fit for it.
    :param cmb_e2e: #if True, use CMB lensing power spectrum from litebird end-to-end simulations.
    :param gnilc: #if True, fit the computed GNILC spectra instead of the full mock data.
    :return results: dictionnary containing A_d, beta_d, T_d, Aw1b, w1bw1b, r and X2red for each (ell,n)
    """
    N,_,Nell=DL.shape
    nparam = len(p0)

    # intitial value for each bin of ell:
    p0L = np.zeros((Nell, nparam))
    for i in range(nparam):
        if np.isscalar(p0[i]) or len(np.atleast_1d(p0[i])) == 1:
            p0L[:, i] = p0[i]  
        else:
            p0L[:, i] = p0[i] 

    #ell array
    b = nmt.NmtBin.from_lmax_linear(lmax=nside*2-1,nlb=Nlbin,is_Dell=True)
    l = b.get_effective_ells()
    l=l[:Nell]
    
    #update keyword for load and save:
    
    kwf = kw+'_fix%s'%fix
    if all_ell:
        kwf = kwf + "_all_ell"
    if iterate:
        kwf = kwf+'_iterate'

    #create folder for parallel    
    if parallel == True:
        pathlib.Path('./best_fits/results_%s_%s.npy'%(kwsave,kwf)).mkdir(parents=True, exist_ok=True)

    # get cmb spectra:
    DL_lensbin, DL_tens= ftl.getDL_cmb(nside=nside,Nlbin=Nlbin,cmb_e2e=cmb_e2e)
    if gnilc:
        DL_lensbin *= 0

    #get frequencies:
    ncross = len(nucross)
    nnus = int((-1 + np.sqrt(ncross * 8 + 1)) / 2.)
    posauto = [int(nnus * i - i * (i + 1) / 2 + i) for i in range(nnus)]
    nu = nucross[posauto]
    freq_pairs = np.array([(i, j) for i in range(nnus) for j in range(i, nnus)])
    nu_i = nu[freq_pairs[:, 0]]
    nu_j = nu[freq_pairs[:, 1]]

    #select function to fit:
    funcfit= eval('ftl.func_'+kw)
     
    if all_ell:
        #put arrays in NcrossxNell shape for all-ell fit
        nu_i = np.tile(nu_i.T, Nell).T
        nu_j = np.tile(nu_j.T, Nell).T
        DL_lensbin= np.repeat(DL_lensbin[:Nell],ncross)
        DL_tens= np.repeat(DL_tens[:Nell],ncross)
        DLdcflat = np.zeros([N,Nell*ncross])
        for i in range(N):
            DLdcflat[i] = np.swapaxes(DL[i,:,:],0,1).flatten()

    if all_ell==False:
        #initialize parameters and chi2:
        paramiterl=np.zeros((Nell,N,nparam))
        chi2l=np.zeros((Nell,N))

        parinfopl =  [{'value':0, 'fixed':0} for i in range(nparam)] #fg params
        parinfopl = np.array([parinfopl for i in range(Nell)])        
        for L in range(Nell):
            parinfopl[L,0]= {'value':p0L[L,0], 'fixed':0,'limited':[1,0],'limits':[0,np.inf]} #Ad
            parinfopl[L,1]= {'value':p0L[L,1], 'fixed':fix,'limited':[1,1],'limits':[0.5,2]} #betad
            parinfopl[L,2]= {'value':1/p0L[L,2], 'fixed':fix,'limited':[1,1],'limits':[1/100,1/3]} #1/Td
            parinfopl[L,3]= {'value':p0L[L,3], 'fixed':0,'limited':[1,0],'limits':[0,np.inf]} #As
            parinfopl[L,4]= {'value':p0L[L,4], 'fixed':fix,'limited':[1,1],'limits':[-5,-2]} #betas    
            if fixr == 1:
                if kw == 'ds_o0':
                    parinfopl[L,6] = {'value': 0, 'fixed': fixr}  # tensor-to-scalar ratio (r)
                elif kw == 'ds_o1bt':
                    parinfopl[L,13] = {'value': 0, 'fixed': fixr}  # tensor-to-scalar ratio (r)
                elif kw == 'ds_o1bts':
                    parinfopl[L,18] = {'value': 0, 'fixed': fixr}  # tensor-to-scalar ratio (r)  
        
        if adaptative:
            res0 = np.load('./best_fits/results_%s_%s.npy'%(kwsave,kwf),allow_pickle=True).item()
            keys = np.array(list(res0.keys()))

            for L in range(Nell):
                if kw == 'ds_o0':
                    raise ValueError('Adaptative is not possible for order 0!')
                
                elif kw == 'ds_o1bt':
                    dust_keys = np.array(keys[6:-2])
                    '''
                    if all(adaptafix(res0[k][L]) == 1 for k in dust_keys):
                        for k in dust_keys:
                            parinfopl[L][np.argwhere(keys==k)[0,0]] = {'value':0, 'fixed':1}*
                    '''

                else:
                    dust_keys = np.concatenate((keys[6:11], keys[13:15]))
                    sync_keys = np.concatenate((keys[11:13], keys[15:-2]))
                    if all(adaptafix(res0[k][L]) == 1 for k in sync_keys):
                        for k in sync_keys:
                            parinfopl[L][np.argwhere(keys==k)[0,0]] = {'value':0, 'fixed':1}
                        '''
                        if all(adaptafix(res0[k][L]) == 1 for k in dust_keys):
                            for k in dust_keys:
                                parinfopl[L][np.argwhere(keys==k)[0,0]] = {'value':0, 'fixed':1}
                        '''

            kwf += '_adaptative'

        #for parallel:
        if parallel:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            perrank = math.ceil(N/size)
            Nmin = rank*perrank
            Nmax = (rank+1)*perrank
        else:
            Nmin = 0
            Nmax = N
        
        #perform the fit

        for n in tqdm(range(Nmin,Nmax)):
            for L in range(Nell):
                # first o1 fit, dust fixed, mom free, r free
                fa = {'x1':nu_i, 'x2':nu_j, 'y':DL[n,:,L], 'err': Linv[L],'ell':L, 'DL_lensbin': DL_lensbin, 'DL_tens': DL_tens,'model_func':funcfit, 'nu0d' : nu0d, 'nu0s' : nu0s}
                m = mpfit(ftl.lkl_mpfit,parinfo= list(parinfopl[L]) ,functkw=fa,quiet=quiet)
                paramiterl[L,n]= m.params
                chi2l[L,n]=m.fnorm/m.dof            
        
        if iterate:
            if kw=='ds_o0':
                raise ValueError('No iteration possible for order 0!')
            elif kw=='ds_o1bt':
                while any(any(np.array([adaptafix(paramiterl[i, :, 6]), adaptafix(paramiterl[i, :, 8])]) == 0) for i in range(Nell)):
                    for n in tqdm(range(Nmin,Nmax)):
                        for L in range(Nell):
                            parinfopl[L][1] = {'value': paramiterl[L,n,1] + paramiterl[L,n,6]/paramiterl[L,n,0] , 'fixed':1}
                            parinfopl[L][2] = {'value': paramiterl[L,n,2] + paramiterl[L,n,8]/paramiterl[L,n,0] , 'fixed':1}
                            fa = {'x1':nu_i, 'x2':nu_j, 'y':DL[n,:,L], 'err': Linv[L],'ell':L, 'DL_lensbin': DL_lensbin, 'DL_tens': DL_tens,'model_func':funcfit, 'nu0d' : nu0d, 'nu0s' : nu0s}
                            m = mpfit(ftl.lkl_mpfit,parinfo= list(parinfopl[L]) ,functkw=fa,quiet=quiet)
                            paramiterl[L,n]= m.params
                            chi2l[L,n]=m.fnorm/m.dof
            else:
                while any(any(np.array([adaptafix(paramiterl[i, :, 6]), adaptafix(paramiterl[i, :, 8]), adaptafix(paramiterl[i, :, 11])]) == 0) for i in range(Nell)):
                    for n in tqdm(range(Nmin,Nmax)):
                        for L in range(Nell):
                            parinfopl[L][1] = {'value': paramiterl[L,n,1] + paramiterl[L,n,6]/paramiterl[L,n,0] , 'fixed':1}
                            parinfopl[L][2] = {'value': paramiterl[L,n,2] + paramiterl[L,n,8]/paramiterl[L,n,0] , 'fixed':1}
                            parinfopl[L][4] = {'value': paramiterl[L,n,4] + paramiterl[L,n,11]/paramiterl[L,n,3], 'fixed':1}
                            fa = {'x1':nu_i, 'x2':nu_j, 'y':DL[n,:,L], 'err': Linv[L],'ell':L, 'DL_lensbin': DL_lensbin, 'DL_tens': DL_tens,'model_func':funcfit, 'nu0d' : nu0d, 'nu0s' : nu0s}
                            m = mpfit(ftl.lkl_mpfit,parinfo= list(parinfopl[L]) ,functkw=fa,quiet=quiet)
                            paramiterl[L,n]= m.params
                            chi2l[L,n]=m.fnorm/m.dof


    
        #return result dictionnary:

        if kw=='ds_o0':
            results={'A_d' : paramiterl[:,:,0], 'beta_d' : paramiterl[:,:,1], 'T_d' : 1/paramiterl[:,:,2], 'A_s': paramiterl[:,:,3], 'beta_s': paramiterl[:,:,4],'A_sd' : paramiterl[:,:,5], 'r':paramiterl[:,:,6], 'X2red': chi2l}
        elif kw=='ds_o1bt':
            results={'A_d' : paramiterl[:,:,0], 'beta_d' : paramiterl[:,:,1], 'T_d' : 1/paramiterl[:,:,2], 'A_s':paramiterl[:,:,3] , 'beta_s':paramiterl[:,:,4], 'A_sd': paramiterl[:,:,5], 'Aw1b' : paramiterl[:,:,6], 'w1bw1b' : paramiterl[:,:,7],'Aw1t' : paramiterl[:,:,8],'w1bw1t' : paramiterl[:,:,9],'w1tw1t' : paramiterl[:,:,10],'Asw1b' : paramiterl[:,:,11],'Asw1t' : paramiterl[:,:,12],'r' : paramiterl[:,:,13], 'X2red': chi2l}
        elif kw=='ds_o1bts':
            results={'A_d' : paramiterl[:,:,0], 'beta_d' : paramiterl[:,:,1], 'T_d' : 1/paramiterl[:,:,2], 'A_s':paramiterl[:,:,3] , 'beta_s':paramiterl[:,:,4], 'A_sd': paramiterl[:,:,5], 'Aw1b' : paramiterl[:,:,6], 'w1bw1b' : paramiterl[:,:,7],'Aw1t' : paramiterl[:,:,8],'w1bw1t' : paramiterl[:,:,9],'w1tw1t' : paramiterl[:,:,10],'Asw1bs' : paramiterl[:,:,11],'w1bsw1bs' : paramiterl[:,:,12],'Asw1b' : paramiterl[:,:,13],'Asw1t' : paramiterl[:,:,14],'Aw1bs' : paramiterl[:,:,15],'w1bw1bs' : paramiterl[:,:,16],'w1tw1bs' : paramiterl[:,:,17],'r' : paramiterl[:,:,18], 'X2red': chi2l}
        else:
            print('unexisting keyword')


    if all_ell:
        funcfit= eval('ftl.func_'+kw+'_all_ell')

        #set initial values:
        parinfopl =  []
        [parinfopl.append({'value':p0L[i,0], 'fixed':0,'limited':[1,0],'limits':[0,np.inf]}) for i in range(Nell)] #A_d
        [parinfopl.append({'value':p0L[i,3], 'fixed':0,'limited':[1,0],'limits':[0,np.inf]}) for i in range(Nell)] #A_s
        [parinfopl.append({'value':p0L[i,5], 'fixed':0,'limited':[0,0],'limits':[-np.inf,np.inf]}) for i in range(Nell)] #A_sd
        if kw=='ds_o1bt' and mompl==False:
            if adaptafix:
                res0=np.load('./best_fits/results_%s_%s.npy'%(kwsave,kwf),allow_pickle=True).item()
                keys= res0.keys()
                #TO DO: write here the update for adaptafix all ell.
            else:
                # kept this part explicit to keep track of the different moment terms:
                [parinfopl.append({'value':0, 'fixed':0,'limited':[0,0],'limits':[-np.inf,np.inf]}) for i in range(Nell)] #Aw1b
                [parinfopl.append({'value':0, 'fixed':0,'limited':[0,0],'limits':[-np.inf,np.inf]}) for i in range(Nell)] #w1bw1b
                [parinfopl.append({'value':0, 'fixed':0,'limited':[0,0],'limits':[-np.inf,np.inf]}) for i in range(Nell)] #Aw1t
                [parinfopl.append({'value':0, 'fixed':0,'limited':[0,0],'limits':[-np.inf,np.inf]}) for i in range(Nell)] #w1bw1t
                [parinfopl.append({'value':0, 'fixed':0,'limited':[0,0],'limits':[-np.inf,np.inf]}) for i in range(Nell)] #w1tw1t
                [parinfopl.append({'value':0, 'fixed':0,'limited':[0,0],'limits':[-np.inf,np.inf]}) for i in range(Nell)] #Asw1b
                [parinfopl.append({'value':0, 'fixed':0,'limited':[0,0],'limits':[-np.inf,np.inf]}) for i in range(Nell)] #Asw1t

        parinfopl.append({'value':p0[1], 'fixed':fix,'limited':[1,1],'limits':[0.5,2]}) #betad
        parinfopl.append({'value':1/p0[2], 'fixed':fix,'limited':[1,1],'limits':[1/100,3]}) #1/Td
        parinfopl.append({'value':p0[4], 'fixed':fix,'limited':[1,1],'limits':[-5,-2]}) #betas    
        parinfopl.append({'value':p0[5], 'fixed':fixr}) # tensor-to-scalar ratio (r) 
        if kw=='ds_o1bt':
            if mompl:
                [parinfopl.append({'value':0,'fixed':0}) for i in range(7)] #moments and power-law indices 
                [parinfopl.append({'value':-0.5,'fixed':0,'limited':[1,1],'limits':[-4,0.1]}) for i in range(7)] #power-law indices 

        elif kw=='ds_o1bts':
            if mompl:
                [parinfopl.append({'value':0,'fixed':0}) for i in range(10)] #moments and power-law indices 
                [parinfopl.append({'value':0,'fixed':0}) for i in range(10)] #power-law indices 
            else:
                raise ValueError('Not implemented yet!')

        #initialize chi2 and best fit array:
        chi2=np.zeros(N)
        paramiter=np.zeros((N,len(parinfopl)))
        
        #for parallel:
        if parallel:
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
            fa = {'x1': nu_i, 'x2': nu_j, 'y': DLdcflat[n], 'err': Linv, 'DL_lensbin': DL_lensbin, 'DL_tens': DL_tens,'ell': np.repeat(l,ncross),'Nell': Nell,'model_func': funcfit, 'nu0d' : nu0d, 'nu0s' : nu0s}
            m = mpfit(ftl.lkl_mpfit, parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiter[n]= m.params
            chi2[n]=m.fnorm/m.dof            
        
        #return result dictionnary:

        if kw=='ds_o0':
            results={'A_d' : np.swapaxes(paramiter[:,:Nell],0,1), 'beta_d' : paramiter[:,3*Nell], 'T_d' : 1/paramiter[:,3*Nell+1], 'A_s': np.swapaxes(paramiter[:,Nell:2*Nell],0,1), 'beta_s': paramiter[:,3*Nell+2],'A_sd' : np.swapaxes(paramiter[:,2*Nell:3*Nell],0,1), 'r':paramiter[:,3*Nell+3], 'X2red': chi2}
        elif kw=='ds_o1bt':
            if mompl:
                maxell = 3*Nell
                results_o0 = {'A_d' : np.swapaxes(paramiter[:,:Nell],0,1), 'beta_d' : paramiter[:,maxell], 'T_d' : 1/paramiter[:,maxell+1], 'A_s':np.swapaxes(paramiter[:,Nell:2*Nell],0,1) , 'beta_s':paramiter[:,maxell+2], 'A_sd':np.swapaxes(paramiter[:,2*Nell:3*Nell],0,1),'r':paramiter[:,maxell+3], 'X2red': chi2}
                results_mom =   {'Aw1b' : paramiter[:,3*Nell+4], 'w1bw1b' : paramiter[:,3*Nell+5],'Aw1t' : paramiter[:,3*Nell+6],'w1bw1t' : paramiter[:,3*Nell+7],'w1tw1t' : paramiter[:,3*Nell+8],'Asw1b' : paramiter[:,3*Nell+9],'Asw1t' : paramiter[:,3*Nell+10]}
                results_mom_pl= {'alpha_Aw1b' : paramiter[:,3*Nell+11], 'alpha_w1bw1b' : paramiter[:,3*Nell+12],'alpha_Aw1t' : paramiter[:,3*Nell+13],'alpha_w1bw1t' : paramiter[:,3*Nell+14],'alpha_w1tw1t' : paramiter[:,3*Nell+15],'alpha_Asw1b' : paramiter[:,3*Nell+16],'alpha_Asw1t' : paramiter[:,3*Nell+17],'r' : paramiter[:,3*Nell+3], 'X2red': chi2}
                results = {**results_o0,**results_mom,**results_mom_pl}

            else:
                maxell = 10*Nell
                results_o0 = {'A_d' : np.swapaxes(paramiter[:,:Nell],0,1), 'beta_d' : paramiter[:,maxell], 'T_d' : 1/paramiter[:,maxell+1], 'A_s':np.swapaxes(paramiter[:,Nell:2*Nell],0,1) , 'beta_s':paramiter[:,maxell+2], 'A_sd':np.swapaxes(paramiter[:,2*Nell:3*Nell],0,1),'r':paramiter[:,maxell+3], 'X2red': chi2}
                results_mom =   {'Aw1b' : np.swapaxes(paramiter[:,3*Nell:4*Nell],0,1), 'w1bw1b' : np.swapaxes(paramiter[:,4*Nell:5*Nell],0,1),'Aw1t' : np.swapaxes(paramiter[:,5*Nell:6*Nell],0,1),'w1bw1t' : np.swapaxes(paramiter[:,6*Nell:7*Nell],0,1),'w1tw1t' : np.swapaxes(paramiter[:,7*Nell:8*Nell],0,1),'Asw1b' : np.swapaxes(paramiter[:,8*Nell:9*Nell],0,1),'Asw1t' : np.swapaxes(paramiter[:,9*Nell:10*Nell],0,1)}
                results = {**results_o0,**results_mom}
        
        elif kw=='ds_o1bts':
            if mompl:
                results_o0 = {'A_d' : np.swapaxes(paramiter[:,:Nell],0,1), 'beta_d' : paramiter[:,3*Nell], 'T_d' : 1/paramiter[:,3*Nell+1], 'A_s':np.swapaxes(paramiter[:,Nell:2*Nell],0,1) , 'beta_s':paramiter[:,3*Nell+2], 'A_sd':np.swapaxes(paramiter[:,2*Nell:3*Nell],0,1)}
                results_mom= {'Aw1b' : paramiter[:,3*Nell+4], 'w1bw1b' : paramiter[:,3*Nell+4],'Aw1t' : paramiter[:,3*Nell+6],'w1bw1t' : paramiter[:,3*Nell+7],'w1tw1t' : paramiter[:,3*Nell+8],'Asw1bs' : paramiter[:,3*Nell+9],'w1bsw1bs' : paramiter[:,3*Nell+10],'Asw1b' : paramiter[:,3*Nell+11],'Asw1t' : paramiter[:,3*Nell+12],'Aw1bs' : paramiter[:,3*Nell+13],'w1bw1bs' : paramiter[:,3*Nell+14],'w1tw1bs' : paramiter[:,3*Nell+15],'r' : paramiter[:,3*Nell+3], 'X2red': chi2}
                results_mom_pl = {}
            results = {**results_o0,**results_mom,**results_mom_pl}
        else:
            raise ValueError('unexisting keyword')

    #save and plot results

    if gnilc:
        kwf += '_gnilc'
    
    if parallel:
        np.save('best_fits/results_%s_%s_p0/res%s.npy'%(kwsave,kwf,rank))    
    else:
        np.save('./best_fits/results_%s_%s.npy'%(kwsave,kwf),results)
        
        if plotres:
            # mask (used to compute theoretical expectations)
            dusttype,synctype,fsky,scale = tuple(str(n) if '.' not in n else float(n) for n in re.findall(r'd(\S+)s(\S+)_(\S+)_scale(\S+\d+)_', kwsave)[0])
            if dusttype == 'b' and synctype == 'b':
                dusttype, synctype = 1, 1
            elif dusttype == 'm' and synctype == 'm':
                dusttype, synctype = 10, 5
            elif dusttype == 'h' and synctype == 'h':
                dusttype, synctype = 12, 7
            else:
                dusttype, synctype = int(dusttype), int(synctype)
            print("dusttype=%s,synctype=%s,fsky=%s,aposcale=%s"%(dusttype,synctype,fsky,scale))
            if fsky==1:
                mask = np.ones(hp.nside2npix(nside))
            elif fsky in ['intersection', 'union']:
                if dusttype == 1 and synctype == 1:
                    complexity = 'baseline'
                elif dusttype == 10 and synctype == 5:
                    complexity = 'medium_complexity'
                elif dusttype == 12 and synctype == 7:
                    complexity = 'high_complexity'
                mask = hp.read_map("./masks/mask_%s_%s_nside%s_aposcale%s.npy"%(fsky,complexity,nside,scale))
            elif 'maskGWD' in kwsave:
                mask = hp.read_map("./masks/mask_GWD_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale))
            else:
                mask = hp.read_map("./masks/mask_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale))
            betabar = np.median(results['beta_d'][~np.isnan(results['X2red'])])
            tempbar = np.median(results['T_d'][~np.isnan(results['X2red'])])
            betasbar= np.median(results['beta_s'][~np.isnan(results['X2red'])])
            fsky = np.mean(mask**2)
            try:
                mom_an = np.load('./analytical_mom/analytical_mom_nside%s_fsky%s_scale%s_Nlbin10_d%ss%s_%s%s%s_%s%s.npy' % (nside, fsky, scale, dusttype, synctype, np.round(betabar,3), np.round(tempbar,3), np.round(betasbar,3),int(nu0d),int(nu0s)), allow_pickle=True).item()
            except:
                print('Computing theoretical expecations for the fitted quantities ...')
                mom_an = anmomlib.getmom(dusttype, synctype, betabar, tempbar, betasbar, mask, Nlbin=Nlbin, nside=nside,nu0d=nu0d,nu0s=nu0s)
                np.save('./analytical_mom/analytical_mom_nside%s_fsky%s_scale%s_Nlbin10_d%ss%s_%s%s%s_%s%s.npy' % (nside, fsky, scale, dusttype, synctype, np.round(betabar,3), np.round(tempbar,3), np.round(betasbar,3),int(nu0d),int(nu0s)), mom_an)
            if all_ell:
                plot_contours=False
            else:
                plot_contours=True
            print('Plotting the results ...')
            ell_pivot_kw = False # theoretical pivot not dependent of ell
            if iterate:
                ell_pivot_kw = True # theoretical pivot dependent of ell if computed iteratively
            if fix==0:
                ell_pivot_kw = True # theoretical pivot dependent of ell if fitted in fix=0
            plib.plotrespdf(l[:Nell],[results],['%s-%s'%(kwsave,kwf)],['darkorange'],mom_an,plot_contours=plot_contours,betadbar=betabar,tempbar=tempbar,betasbar=betasbar,ell_pivot=ell_pivot_kw)
            if all_ell:
                plib.plotr_hist(results,color='darkorange',save=True,kwsave='%s%s'%(kwsave,kwf))
            else:
                plib.plotr_gaussproduct_analytical(results,Nmax=Nell,debug=False,color='darkorange',save=True,kwsave='%s-%s'%(kwsave,kwf))
    return results

# ANALYTICAL LIKELIHOOD MAXIMIZATION

class gauss_like:
    """
    Class for analytical maximization of cross-Cl-based Gaussian likelihood.
    """
    def __init__(self, freq, leff, covmat, comp, beta_d, T_d, beta_s, nu0_d, nu0_s, Dl_lens=None, Dl_tens=None):
        """
        Initialize class by computing mixing matrix A and weight matrix W.
    
        Parameters
        ----------
        freq : array_like
            Array of frequencies at which the mixing matrix should be computed. Can be of shape (Nfreqs,) or (Nfreqs, Ngrid) depending on whether bandpass integration is taken into account.
        leff : array-like
            Effective multipole values at which the mixing matrix has to be computed. Must be of shape (Nbins,).
        covmat : array_like
            Fiducial covariance matrix of dimension (Ncross*Nbins, Ncross*Nbins)
        comp : list
            List of components to be included in the mixing matrix for each bandpower. Each element of comp should be a list of components associated with the considered multipole bin. If len(comp) > Nbins, the last element corresponds to common components to all multipoles.
        beta_d : array_like
            Pivot dust spectral index as a function of multipole bin.
        T_d : array_like
            Pivot dust temperature as a function of multipole bin.
        beta_s : array_like
            Pivot synchrotron spectral index as a function of multipole bin.
        nu0_d : float
            Reference frequency for the polarized dust SED.
        nu0_s : float
            Reference frequency for the polarized synchrotron SED.
        Dl_lens : array_like, optional
            CMB lensing power spectrum in the specified multipole bins. Needed for progressive fits. Default: None.
        Dl_tens : array_like, optional
            CMB tensor modes power spectrum in the specified multipole bins, needed for fitting r globally. Default: None.
    
        Returns
        -------
        None
        """
        self.freq = freq
        self.Nfreqs = len(self.freq)
        self.Ncross = int(self.Nfreqs * (self.Nfreqs+1)/2)
        freq_pairs = np.array([(i, j) for i in range(self.Nfreqs) for j in range(i, self.Nfreqs)])
        self.nu_i = freq[freq_pairs[:, 0]]
        self.nu_j = freq[freq_pairs[:, 1]]

        self.leff = leff
        self.Nbins = len(self.leff)
        self.Dl_lens = Dl_lens
        self.Dl_tens = Dl_tens
        
        self.components = [comp[i].copy() for i in range(len(comp))]
        self.Ncomps = 0
        for i in range(len(comp)):
            self.Ncomps += len(comp[i])
        
        self.beta_d = beta_d * np.ones(self.Nbins)
        self.T_d = T_d * np.ones(self.Nbins)
        self.beta_s = beta_s * np.ones(self.Nbins)
        self.nu0_d, self.nu0_s = nu0_d, nu0_s

        self.covmat = covmat
        self.N_inv = cvl.inverse_covmat(self.covmat, Ncross=self.Ncross, neglect_corbins=False)        
        self.A = self.compute_mixing_matrix()
        self.W = self.compute_weight_matrix()

    def compute_mixing_matrix(self):
        """
        Compute block-diagonal cross-Cl-based mixing matrix given pivot spectral parameters.
    
        Parameters
        ----------
        self
    
        Returns
        -------
        A : array_like
            Computed mixing matrix of shape (Ncross*Nbins, Ncomps)
        """
        A = np.zeros((self.Ncross*self.Nbins, self.Ncomps))
        c_index = 0
        
        for i in range(self.Nbins):
            Nc = len(self.components[i])
            A_ell = np.zeros((self.Ncross, Nc))
            
            for j, c in enumerate(self.components[i]):
                if c == 'cmb':
                    A_ell[:, j] = 1
                else:
                    A_ell[:, j] = self.f_ij(l=i, key=c)#eval('self._'+c)(i)
    
            A[i*self.Ncross : (i+1)*self.Ncross, c_index : c_index+Nc] = A_ell
            c_index += Nc

        if len(self.components) > self.Nbins: # Add common components to the mixing matrix
            for j, c in enumerate(self.components[-1]):
                for i in range(self.Nbins):
                    A[i*self.Ncross : (i+1)*self.Ncross, c_index] = eval('self._'+c)(i)
                c_index += 1
    
        return A

    def compute_weight_matrix(self):
        """
        Compute block-diagonal cross-Cl-based weight matrix given pre-computed mixing matrix.
    
        Parameters
        ----------
        self
    
        Returns
        -------
        W : array_like
            Computed weight matrix of shape (Ncomps*Nbins, Ncross*Nbins)
        """
        W = np.linalg.inv(self.A.T @ self.N_inv @ self.A) @ self.A.T @ self.N_inv
        
        return W

    def maximize(self, data):
        """
        Compute the maximum likelihood estimator of the component amplitudes for the input simulations.

        Parameters
        ----------
        self
        data : array_like
            Input simulations. Must be of shape (N, Ncross, Nbins).

        Returns
        ----------
        results : dict
            Dictionnary of the estimated component amplitudes for the input simulations.
        """
        N = len(data)
        
        moms = []
        self.residuals = np.zeros((N, self.Ncross*self.Nbins))
        chi2r = np.zeros(N)

        for k in range(N):
            moms.append([])
            d = np.concatenate(data[k].T)
            s = self.W @ d
            c_index = 0
            
            for i in range(self.Nbins):
                Nc = len(self.components[i])
                moms[k].append(s[c_index : c_index+Nc])
                c_index += Nc

            if len(self.components) > self.Nbins:
                moms[k].append(s[c_index:])

            res = d - self.A @ s
            self.residuals[k] = res.copy()
            dof = len(d) - len(s)

            chi2r[k] = res.T @ self.N_inv @ res / dof

        keys = []
        for i in range(self.Nbins):
            for key in self.components[i]:
                if key not in keys:
                    keys.append(key)
        
        results = {k: np.zeros((self.Nbins, N)) for k in keys}

        for i in range(self.Nbins):
            c = np.array(self.components[i])
            for key in keys:
                if key in c:
                    for k in range(N):
                        results[key][i, k] = moms[k][i][c==key]
                else:
                    results[key][i] = 0

        results['beta_d'] = self.beta_d
        results['T_d'] = self.T_d
        results['beta_s'] = self.beta_s
        results['chi2r'] = chi2r

        if len(self.components) > self.Nbins:
            if self.components[-1] == ['r']:
                results['r'] = np.zeros(N)
                for k in range(N):
                    results['r'][k] = moms[k][-1]
            else:
                results['A_w1bsw1bs'], results['w1_w1bsw1bs'] = np.zeros((2, N))
                results['A_w1tw1bs'], results['w1_w1tw1bs'] = np.zeros((2, N))
                results['A_w1bw1bs'], results['w1_w1bw1bs'] = np.zeros((2, N))
                for k in range(N):
                    results['A_w1bsw1bs'][k], results['w1_w1bsw1bs'][k], results['A_w1tw1bs'][k], results['w1_w1tw1bs'][k], results['A_w1bw1bs'][k], results['w1_w1bw1bs'][k] = moms[k][-1]#, results['A_w1tw1bs'], results['w1_w1tw1bs'], results['A_w1bw1bs'], results['w1_w1bw1bs'] = moms[k][-1]
                    
                results['gamma_w1bsw1bs'] = self.gamma_w1bsw1bs
                results['gamma_w1tw1bs'] = self.gamma_w1tw1bs
                results['gamma_w1bw1bs'] = self.gamma_w1bw1bs

        if self.Dl_lens is not None:
            total_cmb = results['cmb'].T
            cov_cmb = np.cov(results['cmb'])
            N_inv_cmb = np.linalg.inv(cov_cmb)
    
            A_cmb = np.zeros((self.Nbins, 1))
            A_cmb[:, 0] = self.Dl_tens
    
            W_cmb = np.linalg.inv(A_cmb.T @ N_inv_cmb @ A_cmb) @ A_cmb.T @ N_inv_cmb
    
            samp_nomarg = np.zeros((N, 2))
            chi2r_nomarg = np.zeros(N)
    
            for i in range(N):
                d_cmb = total_cmb[i] - self.Dl_lens
                s_cmb = W_cmb @ d_cmb
        
                res = d_cmb - A_cmb @ s_cmb
                dof = len(d_cmb) - len(s_cmb)
        
                samp_nomarg[i, 0] = s_cmb[0]
                chi2r_nomarg[i] = res.T @ N_inv_cmb @ res / dof
    
            samp_nomarg[:, 1] = 100
    
            results['r'] = samp_nomarg[:, 0]

        return results

    def run(self, data, n_iter=3, adaptative=True, progressive=False, pl_moms=False, HVTWD=False, PCA=False, Azzoni=False, OMP=False, GS=False):
        """
        Run component separation for the input simulations.

        Parameters
        ----------
        self
        data : array_like
            Input simulations. Must be of shape (N, Ncross, Nbins).
        n_iter : int, optional
            Number of iterations to run to find ideal pivot values. Default: 3.
        adaptative : bool, optional
            Whether to re-run the component separation after deleting the undetected moments. Default: True.
        progressive : bool, optional
            Progressive version of adaptative where insignificant moments are removed one by one. Default: False.
        pl_moms : bool, optional
            Whether to re-run component separation using a power law of ell parametrization for w1bsw1bs, w1tw1bs, and w1bw1bs. Default: False.
        HVTWD : bool, optional
            Whether to reiterate the fit for the CMB component after smoothing the fitted moments. Defalut: False.
        PCA : bool, optional
            Perform a principal component analysis (PCA) after iterative estimate of the mixing matrix. Default: False.
        Azzoni : bool, optional
            Re-iterate the fit à la Azzoni et al. Default: False.
        OMP : bool, optional
            Pipeline for orthogonal matching pursuit. Default: False.
        GS : bool, optional
            Pipeline for Gram-Schmidt orthogonalization

        Returns
        ----------
        results : dict
            Dictionnary of the estimated component amplitudes for the input simulations.
        """
        if GS:
            N = len(data)
            
            for i in trange(n_iter, desc='Iterations'):
                results = self.maximize(data)
                
                self.beta_d += np.mean(results['Aw1b'] / results['A'], axis=1)
                self.T_d = 1 / (1/self.T_d + np.mean(results['Aw1t'] / results['A'], axis=1))
                self.beta_s += np.mean(results['Asw1bs'] / results['As'], axis=1)
                
                self.beta_d = np.clip(self.beta_d, 1, 2)
                self.T_d = np.clip(self.T_d, 15, 25)
                self.beta_s = np.clip(self.beta_s, -4, -2)
    
                self.A = self.compute_mixing_matrix()
                self.W = self.compute_weight_matrix()

            print('Fit simulations using updated pivot values...')

            comp = ['cmb', 'A', 'As', 'Asd']
            
            for i in range(self.Nbins):
                self.components[i] = comp.copy()

            moms = np.array(['A', 'As', 'w1b', 'w1t', 'w1bs',
                             #'w2b', 'w2t', 'w2bt', 'w2bs',
                             #'w3b', 'w3t', 'w3bbt', 'w3btt', 'w3bs',
                             #'w4b', 'w4t', 'w4bbbt', 'w4bbtt', 'w4bttt', 'w4bs',
                            ])
            
            DLmoms = []
            for i, k1 in enumerate(moms):
                for j, k2 in enumerate(moms[i:]):
                    if not(k1[0] == 'A' and k2[0] == 'A'):
                        k = f'{k1}{k2}'
                        if k not in comp:
                            DLmoms.append(k)

            self.Ncomps = (len(comp) + len(DLmoms)) * self.Nbins
            self.A = np.zeros((self.Ncross*self.Nbins, self.Ncomps))
            c_index = 0
            
            for i in range(self.Nbins):
                cov = self.covmat[i*self.Ncross:(i+1)*self.Ncross, i*self.Ncross:(i+1)*self.Ncross]
                N_inv = cvl.inverse_covmat(cov, Ncross=self.Ncross, neglect_corbins=False)

                A_ell = np.zeros((self.Ncross, len(comp)+len(DLmoms)))

                for j, c in enumerate(comp):
                    if c == 'cmb':
                        A_ell[:, j] = 1
                    else:
                        A_ell[:, j] = self.f_ij(i, key=c)

                Nc = len(comp)
                for j, c in enumerate(DLmoms):
                    A_ell[:, j+len(comp)] = self.f_ij(i, key=c)
                    self.components[i].append(f'X{j+1}')

                    for k in range(Nc):
                        A_ell[:, j+len(comp)] -= (A_ell[:, j+len(comp)].T @ N_inv @ A_ell[:, k]) / (A_ell[:, k].T @ N_inv @ A_ell[:, k]) * A_ell[:, k]

                    Nc += 1

                self.A[i*self.Ncross : (i+1)*self.Ncross, c_index : c_index+Nc] = A_ell.copy()
                c_index += Nc

            self.W = self.compute_weight_matrix()

            results = self.maximize(data)

            return results
            
        
        if OMP:
            N = len(data)
            
            for i in trange(n_iter, desc='Iterations'):
                results = self.maximize(data)
                
                self.beta_d += np.mean(results['Aw1b'] / results['A'], axis=1)
                self.T_d = 1 / (1/self.T_d + np.mean(results['Aw1t'] / results['A'], axis=1))
                self.beta_s += np.mean(results['Asw1bs'] / results['As'], axis=1)
                
                self.beta_d = np.clip(self.beta_d, 1, 2)
                self.T_d = np.clip(self.T_d, 15, 25)
                self.beta_s = np.clip(self.beta_s, -4, -2)
    
                self.A = self.compute_mixing_matrix()
                self.W = self.compute_weight_matrix()

            print('Fit simulations using updated pivot values...')

            comp = ['cmb', 'A', 'As', 'Asd']
            
            for i in range(self.Nbins):
                self.components[i] = comp.copy()
            self.Ncomps = len(comp) * self.Nbins

            moms = np.array(['A', 'As', 'w1b', 'w1t', 'w1bs',
                             #'w2b', 'w2t', 'w2bt', 'w2bs',
                             #'w3b', 'w3t', 'w3bbt', 'w3btt', 'w3bs',
                             #'w4b', 'w4t', 'w4bbbt', 'w4bbtt', 'w4bttt', 'w4bs',
                            ])
            
            DLmoms = []
            for i, k1 in enumerate(moms):
                for j, k2 in enumerate(moms[i:]):
                    if not(k1[0] == 'A' and k2[0] == 'A'):
                        k = f'{k1}{k2}'
                        if k not in comp:
                            DLmoms.append(k)

            for i in range(self.Nbins):
                gauss = gauss_like(self.freq, np.array([self.leff[i]]), self.covmat[i*self.Ncross:(i+1)*self.Ncross, i*self.Ncross:(i+1)*self.Ncross], [comp], self.beta_d[i], self.T_d[i], self.beta_s[i], self.nu0_d, self.nu0_s)
                res = gauss.maximize(data[:, :, i, None])
                
                atoms = DLmoms.copy()
                p_max = np.inf
                threshold = 1

                while p_max > threshold:
                    p = np.zeros((len(atoms), N))
                    for j, c in enumerate(atoms):
                        a = self.f_ij(i, key=c)

                        for k in range(N):
                            p[j, k] = np.abs((a.T @ gauss.N_inv @ gauss.residuals[k]) / np.sqrt(a.T @ gauss.N_inv @ a))
                    
                    p = np.mean(p, axis=1)
                        
                    idx_max = np.argmax(p)
                    p_max = p[idx_max]
                    print(i, atoms[idx_max], p_max)

                    if p_max > threshold:
                        self.components[i].append(atoms[idx_max])
                        self.Ncomps += 1
                            
                        gauss.components[0].append(atoms[idx_max])
                        gauss.Ncomps += 1

                        gauss.A = gauss.compute_mixing_matrix()
                        gauss.W = gauss.compute_weight_matrix()
                            
                        res = gauss.maximize(data[:, :, i, None]) 
                            
                        atoms.remove(atoms[idx_max])

            self.A = self.compute_mixing_matrix()
            self.W = self.compute_weight_matrix()

            results = self.maximize(data)
            
            return results

            """
            atoms = [DLmoms.copy() for i in range(self.Nbins)]

            for it in range(100):
                score = 0
                
                for i in range(self.Nbins):
                    for j, c in enumerate(atoms[i]):
                        a = np.zeros(self.Ncross*self.Nbins)
                        a[i*self.Ncross : (i+1)*self.Ncross] = self.f_ij(i, key=c)
                        
                        p = np.zeros(N)
                        for k in range(N):
                            p[k] = np.abs((a.T @ self.N_inv @ self.residuals[k]) / np.sqrt(a.T @ self.N_inv @ a))

                        print(i, c, np.mean(p))
                        if np.mean(p) > score:
                            score = np.mean(p)
                            idx_bin, comp = i, c

                self.components[idx_bin].append(comp)
                self.Ncomps += 1
                atoms[idx_bin].remove(comp)

                self.A = self.compute_mixing_matrix()
                self.W = self.compute_weight_matrix()

                results = self.maximize(data)
                print(np.mean(results['chi2r']))

            return results
            """

        
        for i in trange(n_iter, desc='Iterations'):
            results = self.maximize(data)
            
            self.beta_d += np.mean(results['Aw1b'] / results['A'], axis=1)
            self.T_d = 1 / (1/self.T_d + np.mean(results['Aw1t'] / results['A'], axis=1))
            #self.T_d += np.mean(results['Aw1t'] / results['A'], axis=1)
            #self.T_d = np.exp(np.log(self.T_d) + np.mean(results['Aw1t'] / results['A'], axis=1))
            self.beta_s += np.mean(results['Asw1bs'] / results['As'], axis=1)
            
            self.beta_d = np.clip(self.beta_d, 1, 2)
            self.T_d = np.clip(self.T_d, 15, 25)
            self.beta_s = np.clip(self.beta_s, -4, -2)

            self.A = self.compute_mixing_matrix()
            self.W = self.compute_weight_matrix()
        
        print('Fit simulations using updated pivot values...')
        
        if n_iter > 0:
            #self.beta_d[:] = np.mean(self.beta_d)
            #self.T_d[:] = np.mean(self.T_d)
            #self.beta_s[:] = np.mean(self.beta_s)
            
            for i in range(self.Nbins):
                #self.components[i].remove('Aw1b')
                #self.components[i].remove('Aw1t')
                #self.components[i].remove('Asw1bs')
                bonus = 'w1bw3b'
                to_append = ['Aw2b', 'Aw2t', 'Asw2bs',
                            #'w1bw2b', 'w1bw2t', 'w1tw2b', 'w1tw2t']
                            #'w2bw2b', 'w2tw2t', 'w2bw2t', 'w2bsw2bs']
                            bonus]
                            #]
                
                if not adaptative:
                    to_append = []

                for comp in to_append:
                    self.components[i].append(comp)
                
                self.Ncomps += len(to_append) - 0
            
            self.A = self.compute_mixing_matrix()
            self.W = self.compute_weight_matrix()

        results = self.maximize(data)
        
        if adaptative:
            print('Run adaptative fits...')
            """
            dust_keys = ['Aw1b', 'Aw1t', 'w1bw1b', 'w1tw1t', 'w1bw1t', 'Asw1b', 'Asw1t']
            sync_keys = ['Asw1bs', 'w1bsw1bs', 'Aw1bs', 'w1bw1bs', 'w1tw1bs']

            for i in range(self.Nbins):
                if all(np.mean(results[k][i]) / np.std(results[k][i]) < 0.05 for k in sync_keys):
                    for k in sync_keys:
                        self.components[i].remove(k)
                        self.Ncomps -= 1
                        
                    if all(np.mean(results[k][i]) / np.std(results[k][i]) < 0.05 for k in dust_keys):
                        for k in dust_keys:
                            self.components[i].remove(k)
                            self.Ncomps -= 1
            
            #keys = ['Aw1b', 'Aw1t', 'w1bw1b', 'w1tw1t', 'w1bw1t', 'Asw1bs', 'w1bsw1bs', 'Asw1b', 'Asw1t', 'Aw1bs', 'w1bw1bs', 'w1tw1bs']
            keys = ['Aw1b', 'Aw1t', 'Asw1bs', 'w1bw1b', 'Asw1t', 'w2bw2b']
            for i in range(self.Nbins):
                for k in keys:
                    if np.mean(results[k][i]) / np.std(results[k][i]) < 0.01:
                        self.components[i].remove(k)
                        self.Ncomps -= 1
            """
            """
            o1d_keys = ['Aw1b', 'Aw1t']
            o2d_keys = ['w1bw1b', 'w1tw1t', 'w1bw1t']
            o1s_keys = ['Asw1bs']
            o2s_keys = ['w1bsw1bs']
            o1ds_keys1 = ['Asw1b', 'Asw1t']
            o1ds_keys2 = ['Aw1bs']
            o2ds_keys = ['w1bw1bs', 'w1tw1bs']
            
            ob = ['Aw1b', 'Asw1b', 'w1bw1b']
            ot = ['Aw1t', 'Asw1t', 'w1tw1t']
            os = ['Asw1bs', 'Aw1bs', 'w1bsw1bs']
            obt = ['w1bw1t']
            obs = ['w1bw1bs']
            ots = ['w1tw1bs']

            o1b = ['Aw1b', 'Asw1b']
            o2b= ['w1bw1b']
            o1t = ['Aw1t', 'Asw1t']
            o2t = ['w1tw1t']
            o1s = ['Asw1bs', 'Aw1bs']
            o2s = ['w1bsw1bs']
            o2bt = ['w1bw1t']
            o2bs = ['w1bw1bs']
            o2ts = ['w1tw1bs']
            """
            N = len(data)
            
            for i in range(self.Nbins):
                """
                if np.all(np.array([np.abs(np.mean(results[k]i]) / np.std(results[k][i])) for k in ['Aw1b', 'Aw1t']]) < 2 / np.sqrt(N)):
                    self.components[i].remove('Aw1b')
                    self.components[i].remove('Aw1t')
                    self.Ncomps -= 2
                    
                if np.abs(np.mean(results['Asw1bs'][i]) / np.std(results['Asw1bs'][i])) < 2 / np.sqrt(N):
                    self.components[i].remove('Asw1bs')
                    self.Ncomps -= 1
                """
                """
                if np.any(np.array([np.abs(np.mean(results[k][i]) / np.std(results[k][i])) for k in ['w1tw1t', 'w1bw1t']]) < 2 / np.sqrt(N)):
                    if np.abs(np.mean(results['w1tw1t'][i]) / np.std(results['w1tw1t'][i])) < np.abs(np.mean(results['w1bw1t'][i]) / np.std(results['w1bw1t'][i])):
                        self.components[i].remove('w1tw1t')
                    else:
                        self.components[i].remove('w1bw1t')
                            
                    self.Ncomps -= 1
                """
                
                for k in np.array(self.components[i]):
                    #if k not in ['A', 'As', 'Asd', 'Aw1b', 'Aw1t', 'Asw1bs', 'w1bw1b', 'cmb']:
                    if k not in ['A', 'As', 'Asd', 'w1bw1b', 'w1tw1t', 'cmb']:
                        if np.abs(np.mean(results[k][i]) / np.std(results[k][i])) < 2 / np.sqrt(N):
                            self.components[i].remove(k)
                            self.Ncomps -= 1
    
            self.A = self.compute_mixing_matrix()
            self.W = self.compute_weight_matrix()
    
            results = self.maximize(data)
            
            """
            for i in range(self.Nbins):
                if np.mean(results['w1tw1t'][i]) == 0 and np.abs(np.mean(results['w1bw1t'][i]) / np.std(results['w1bw1t'][i])) < 2 / np.sqrt(N):
                    self.components[i].remove('w1bw1t')
                    self.Ncomps -= 1

                elif np.mean(results['w1bw1t'][i]) == 0 and np.abs(np.mean(results['w1tw1t'][i]) / np.std(results['w1tw1t'][i])) < 2 / np.sqrt(N):
                    self.components[i].remove('w1tw1t')
                    self.Ncomps -= 1
            """
            for i in range(self.Nbins):
                if 'w1tw1t' in self.components[i] and np.abs(np.mean(results['w1tw1t'][i]) / np.std(results['w1tw1t'][i])) < 2 / np.sqrt(N):
                    self.components[i].remove('w1tw1t')
                    self.Ncomps -= 1

            self.A = self.compute_mixing_matrix()
            self.W = self.compute_weight_matrix()
    
            results = self.maximize(data)

            for i in range(self.Nbins):
                if 'w1bw1b' in self.components[i] and np.abs(np.mean(results['w1bw1b'][i]) / np.std(results['w1bw1b'][i])) < 2 / np.sqrt(N):
                    self.components[i].remove('w1bw1b')
                    self.Ncomps -= 1
            
            self.A = self.compute_mixing_matrix()
            self.W = self.compute_weight_matrix()
    
            results = self.maximize(data)

        if progressive:
            #peak_ini = np.mean(results['r'])
            #sigma_ini = np.std(results['r'])
            for i in trange(self.Nbins, desc='Adaptative fits'):
                peak_ini = np.mean(results['cmb'][i])
                sigma_ini = np.std(results['cmb'][i])
                
                cov = np.cov(np.array([results[k][i] if k != 'r' else results[k] for k in np.concatenate((self.components[i], ['r']))]))
                sig = np.sqrt(np.diag(cov))
                cor = cov / np.outer(sig, sig)

                stop = False
                while not stop and len(self.components[i]) > 5:
                    components = [self.components[i].copy() for i in range(self.Nbins)]
                    Ncomps = self.Ncomps
                
                    score = np.zeros(len(self.components[i]))
                    for j, k in enumerate(self.components[i]):
                        if k not in ['A', 'As', 'Asd', 'cmb', 'w1bw1b']:
                            score[j] = np.abs(np.std(results[k][i]) / np.mean(results[k][i]) )#/ cor[np.where(np.array(self.components[i]) == k)[0][0], -1])
    
                    to_remove = self.components[i][np.argmax(score)]
                    self.components[i].remove(to_remove)
                    self.Ncomps -= 1

                    self.A = self.compute_mixing_matrix()
                    self.W = self.compute_weight_matrix()
                    res = self.maximize(data)

                    if np.abs(np.mean(res['cmb'][i]) - np.mean(results['cmb'][i])) / np.std(results['cmb'][i]) < 0.2:
                        #if np.abs(np.mean(res['r']) - np.mean(results['r'])) / np.std(results['r']) < 0.2:
                        #if np.abs(np.mean(res['r']) - peak_ini) / sigma_ini < 0.1:
                        results = res
                        #print(f'did remove {to_remove}')

                    else:
                        self.components = [components[i].copy() for i in range(self.Nbins)]
                        self.Ncomps = Ncomps
                        stop = True
                        #print(f'did NOT remove {to_remove}')

                    self.res = res
                    self.score = score

            self.A = self.compute_mixing_matrix()
            self.W = self.compute_weight_matrix()

            results = self.maximize(data)
                
        if pl_moms:
            for i in range(self.Nbins):
                for k in ['w1bsw1bs', 'w1tw1bs', 'w1bw1bs']:
                    self.components[i].remove(k)
                    self.Ncomps -= 1

            self.components.append(['A_w1bsw1bs', 'w1_w1bsw1bs', 'A_w1tw1bs', 'w1_w1tw1bs', 'A_w1bw1bs', 'w1_w1bw1bs'])
            self.Ncomps += len(self.components[-1])
            self.gamma_w1bsw1bs = 0
            self.gamma_w1tw1bs = 0
            self.gamma_w1bw1bs = 0

            for i in trange(10, desc='Power law fits'):
                self.A = self.compute_mixing_matrix()
                self.W = self.compute_weight_matrix()

                results = self.maximize(data)
                self.gamma_w1bsw1bs += np.mean(results['w1_w1bsw1bs'] / results['A_w1bsw1bs'])
                self.gamma_w1tw1bs += np.mean(results['w1_w1tw1bs'] / results['A_w1tw1bs'])
                self.gamma_w1bw1bs += np.mean(results['w1_w1bw1bs'] / results['A_w1bw1bs'])  

        if HVTWD:
            print('Run HVTWD...')
            
            N = len(data)
            keys = list(results)
            Nmoms = 0
            
            for par in keys:
                if par not in ['A', 'As', 'Asd', 'cmb', 'beta_d', 'T_d', 'beta_s', 'chi2r']:
                    Nmoms += 1
                    for k in range(N):
                        results[par][:, k] = scipy.ndimage.gaussian_filter(results[par][:, k], sigma=10, mode='constant')

            s_fg = np.zeros((N, (Nmoms+3)*self.Nbins))
            for k in range(N):
                idx = 0
                for par in keys:
                    if par not in ['cmb', 'beta_d', 'T_d', 'beta_s', 'chi2r']:
                        s_fg[k, np.arange(self.Nbins)*Nmoms + idx] = results[par][:, k]
                        idx += 1

            for i in range(self.Nbins):
                self.components[i].remove('cmb')
                self.Ncomps -= 1
            A_fg = self.compute_mixing_matrix()

            data_cleaned = np.zeros_like(data)
            for k in range(N):
                data_cleaned[k] = (np.concatenate(data[k].T) - A_fg @ s_fg[k]).reshape((self.Nbins, self.Ncross)).T

            self.components = [['cmb'] for i in range(self.Nbins)]
            self.Ncomps = self.Nbins
            self.A = self.compute_mixing_matrix()
            self.W = self.compute_weight_matrix()

            results = self.maximize(data_cleaned)
             
        if PCA:
            print('Run PCA...')
            
            N = len(data)

            self.Npc = np.zeros(self.Nbins, dtype=np.int64)            
            
            moms = np.array(['A', 'As', 'w1b', 'w1t', 'w1bs',
                             'w2b', 'w2t', 'w2bt', 'w2bs',
                             'w3b', 'w3t', 'w3bbt', 'w3btt', 'w3bs',
                             'w4b', 'w4t', 'w4bbbt', 'w4bbtt', 'w4bttt', 'w4bs',
                            ])
            
            comp = ['cmb', 'A', 'As', 'Asd', 'w1bw1b']
            Nfix = len(comp)

            for i in range(self.Nbins):
                self.components[i] = comp.copy()
            self.Ncomps = Nfix * self.Nbins
            
            comp_tot = comp.copy()
            
            for i, k1 in enumerate(moms):
                for j, k2 in enumerate(moms[i:]):
                    if not(k1[0] == 'A' and k2[0] == 'A'):
                        key = f'{k1}{k2}'
                        if key not in comp_tot:
                            comp_tot.append(f'{k1}{k2}')
                        
            idx_fix = np.arange(Nfix)
            idx_pca = np.arange(Nfix, len(comp_tot))

            self.eigenvals_sorted = np.zeros((self.Nbins, len(comp_tot)-len(idx_fix)))
            self.eigenvects_sorted = np.zeros((self.Nbins, len(comp_tot)-len(idx_fix), len(comp_tot)-len(idx_fix)))

            A = []
            
            for i in range(self.Nbins):
                # Create orthogonal atoms in bin i
                gauss = gauss_like(self.freq, np.array([self.leff[i]]), self.covmat[i*self.Ncross:(i+1)*self.Ncross, i*self.Ncross:(i+1)*self.Ncross], [comp_tot], self.beta_d[i], self.T_d[i], self.beta_s[i], self.nu0_d, self.nu0_s)
                
                F = gauss.A.T @ gauss.N_inv @ gauss.A

                F_ff = F[np.ix_(idx_fix, idx_fix)]
                F_fp = F[np.ix_(idx_fix, idx_pca)]
                F_pf = F_fp.T
                F_pp = F[np.ix_(idx_pca, idx_pca)]

                F_ell = F_pp - F_pf @ np.linalg.inv(F_ff) @ F_fp
                F_ell = (F_ell + F_ell.T) / 2
                
                eigenvals, eigenvects = np.linalg.eigh(F_ell)
                sort = np.argsort(eigenvals)[::-1]
                self.eigenvals_sorted[i] = eigenvals[sort]
                self.eigenvects_sorted[i] = eigenvects[:, sort]

                atoms = list((gauss.A[:, idx_pca] @ self.eigenvects_sorted[i]).T)

                # Add atoms iteratively
                gauss = gauss_like(self.freq, np.array([self.leff[i]]), self.covmat[i*self.Ncross:(i+1)*self.Ncross, i*self.Ncross:(i+1)*self.Ncross], [comp], self.beta_d[i], self.T_d[i], self.beta_s[i], self.nu0_d, self.nu0_s)
                res = gauss.maximize(data[:, :, i, None])

                p_max = np.inf
                threshold = 0.2
                idx_added = []

                while p_max > threshold:
                    p = np.zeros((len(atoms), N))
                    for j, a in enumerate(atoms):
                        if j not in idx_added:
                            for k in range(N):
                                p[j, k] = np.abs((a.T @ gauss.N_inv @ gauss.residuals[k]) / np.sqrt(a.T @ gauss.N_inv @ a))

                    p = np.mean(p, axis=1)

                    idx_max = np.argmax(p)
                    p_max = p[idx_max]
                    print(i, f'X{idx_max+1}', p_max)

                    if p_max > threshold:
                        self.components[i].append(f'X{idx_max+1}')
                        self.Ncomps += 1
                        
                        gauss.components[0].append(f'X{idx_max+1}')
                        gauss.Ncomps += 1

                        A_ell = np.zeros((gauss.Ncross, gauss.Ncomps))
                        A_ell[:, :gauss.Ncomps-1] = gauss.A.copy()
                        A_ell[:, -1] = atoms[idx_max].copy()

                        gauss.A = A_ell.copy()
                        gauss.W = gauss.compute_weight_matrix()

                        res = gauss.maximize(data[:, :, i, None])

                        idx_added.append(idx_max)

                A.append(gauss.A.copy())

            self.A = np.zeros((self.Ncross*self.Nbins, self.Ncomps))
            
            Nc = 0
            for i in range(self.Nbins):
                dNc = len(A[i].T)
                self.A[i*self.Ncross : (i+1)*self.Ncross, Nc : Nc+dNc] = A[i].copy()
                Nc += dNc

            self.W = self.compute_weight_matrix()

            results = self.maximize(data)
            
            """
                self.Npc[i] = len(comp) - len(idx_fix)
                self.Npc[i] = np.argmax(np.cumsum(self.eigenvals_sorted[i]) / np.sum(self.eigenvals_sorted[i]) > 0.99)
            
            self.Ncomps = np.sum(self.Npc + len(idx_fix))
            self.A = np.zeros((self.Ncross*self.Nbins, self.Ncomps))
            self.components = [list(comp[idx_fix])
                               for j in range(self.Nbins)]

            c_index = 0
            for i in range(self.Nbins):
                A_ell = np.zeros((self.Ncross, self.Npc[i]+len(idx_fix)))

                for j, idx in enumerate(idx_fix):
                    if comp[idx] == 'cmb':
                        A_ell[:, j] = 1
                    else:
                        A_ell[:, j] = self.f_ij(l=i, key=comp[idx])

                A_pca = gauss[i].A[:, idx_pca] @ self.eigenvects_sorted[i]
                
                for j in range(self.Npc[i]):
                    self.components[i].append(f'X{j+1}')
                    A_ell[:, j+len(idx_fix)] = A_pca[:, j].copy()

                self.A[i*self.Ncross : (i+1)*self.Ncross, c_index : c_index+self.Npc[i]+len(idx_fix)] = A_ell.copy()
                c_index += self.Npc[i] + len(idx_fix)

            self.W = self.compute_weight_matrix()
            results = self.maximize(data)
            """
        
        if PCA and 0==1:
            print('Run PCA...')
            """
            self.Npc = np.zeros(self.Nbins, dtype=np.int64)
            comp = [['A', 'As', 'Asd', 'Aw1b', 'Aw1t', 'Asw1bs', 'w1bw1b', 'w1tw1t', 'w1bw1t', 'w1bsw1bs', 'Asw1b', 'Asw1t', 'Aw1bs', 'w1bw1bs', 'w1tw1bs']]

            self.eigenvals_sorted = np.zeros((self.Nbins, len(comp[0])))
            self.eigenvects_sorted = np.zeros((self.Nbins, len(comp[0]),  len(comp[0])))
            
            for i in range(self.Nbins):
                gauss_ell = gauss_like(self.freq, np.array([self.leff[i]]), self.covmat[i*self.Ncross:(i+1)*self.Ncross, i*self.Ncross:(i+1)*self.Ncross], comp, self.beta_d, self.T_d, self.beta_s, self.nu0_d, self.nu0_s)
                
                F_ell = gauss_ell.A.T @ gauss_ell.N_inv @ gauss_ell.A
                eigenvals, eigenvects = np.linalg.eigh(F_ell)
                sort = np.argsort(eigenvals)[::-1]
                self.eigenvals_sorted[i] = eigenvals[sort]
                self.eigenvects_sorted[i] = eigenvects[:, sort]

                #cum = np.cumsum(self.eigenvals_sorted[i]) / np.sum(self.eigenvals_sorted[i])
                #self.Npc[i] = np.searchsorted(cum, 0.99) + 1
                self.Npc[i] = len(comp[0])

            self.Ncomps = np.sum(self.Npc+0)
            self.A = np.zeros((self.Ncross*self.Nbins, self.Ncomps))
            self.components = [[]
                               for i in range(self.Nbins)]

            c_index = 0
            for i in range(self.Nbins):    
                A_ell = np.zeros((self.Ncross, self.Npc[i]+0))

                for j in range(self.Npc[i]):
                    self.components[i].append(f'X{j+1}')
                    for k, c in enumerate(comp[0]):
                        A_ell[:, j+0] += self.eigenvects_sorted[i, k, j] * eval('self._'+c)(i)

                self.A[i*self.Ncross : (i+1)*self.Ncross, c_index : c_index+self.Npc[i]+0] = A_ell
                c_index += self.Npc[i] + 0
            """
            
            self.Npc = np.zeros(self.Nbins, dtype=np.int64)
            comp = np.array(['cmb', 'A', 'As', 'Asd', 'Aw1b', 'Aw1t', 'Asw1bs', 'w1bw1b', 'w1tw1t', 'w1bw1t', 'w1bsw1bs', 'Asw1b', 'Asw1t', 'Aw1bs', 'w1bw1bs', 'w1tw1bs'])
            #comp = np.array(['cmb', 'A', 'As', 'Asd', 'Aw1b', 'Aw1t', 'Asw1bs', 'w1bw1b', 'w1tw1t', 'w1bw1t', 'w1bsw1bs', 'Asw2bs'])
            #comp = np.array(['cmb', 'A', 'As', 'Asd', 'Aw1b', 'Aw1t', 'Asw1bs', 'w1bw1b', 'w1tw1t', 'w1bw1t', 'w1bsw1bs', 'Asw1b', 'Asw1t', 'Aw1bs', 'w1bw1bs', 'w1tw1bs', 'Aw2b', 'Aw2t', 'Asw2bs', 'w1bw3b'])
            idx_fix = np.arange(8)
            idx_pca = np.arange(8, len(comp))

            self.eigenvals_sorted = np.zeros((self.Nbins, len(comp)-len(idx_fix)))
            self.eigenvects_sorted = np.zeros((self.Nbins, len(comp)-len(idx_fix),  len(comp)-len(idx_fix)))

            gauss = []
            for i in range(self.Nbins):
                gauss_ell = gauss_like(self.freq, np.array([self.leff[i]]), self.covmat[i*self.Ncross:(i+1)*self.Ncross, i*self.Ncross:(i+1)*self.Ncross], [list(comp)], self.beta_d, self.T_d, self.beta_s, self.nu0_d, self.nu0_s)
                gauss.append(gauss_ell)
                
                F = gauss_ell.A.T @ gauss_ell.N_inv @ gauss_ell.A

                F_ff = F[np.ix_(idx_fix, idx_fix)]
                F_fp = F[np.ix_(idx_fix, idx_pca)]
                F_pf = F_fp.T
                F_pp = F[np.ix_(idx_pca, idx_pca)]

                F_ell = F_pp - F_pf @ np.linalg.inv(F_ff) @ F_fp
                F_ell = 0.5 * (F_ell + F_ell.T)
                
                eigenvals, eigenvects = np.linalg.eigh(F_ell)
                sort = np.argsort(eigenvals)[::-1]
                self.eigenvals_sorted[i] = eigenvals[sort]
                self.eigenvects_sorted[i] = eigenvects[:, sort]

                self.Npc[i] = len(comp) - len(idx_fix)

            self.Ncomps = np.sum(self.Npc + len(idx_fix))
            self.A = np.zeros((self.Ncross*self.Nbins, self.Ncomps))
            self.components = [list(comp[idx_fix])
                               for j in range(self.Nbins)]

            c_index = 0
            for i in range(self.Nbins):              
                A_ell = np.zeros((self.Ncross, self.Npc[i]+len(idx_fix)))
                for j, idx in enumerate(idx_fix):
                    if comp[idx] == 'cmb':
                        A_ell[:, j] = 1
                    else:
                        A_ell[:, j] = self.f_ij(l=i, key=comp[idx])#eval('self._'+comp[idx])(i)

                A_pca = gauss[i].A[:, idx_pca] @ self.eigenvects_sorted[i]
                for j in range(self.Npc[i]):
                    self.components[i].append(f'X{j+1}')
                    A_ell[:, j+len(idx_fix)] = A_pca[:, j]

                self.A[i*self.Ncross : (i+1)*self.Ncross, c_index : c_index+self.Npc[i]+len(idx_fix)] = A_ell
                c_index += self.Npc[i] + len(idx_fix)
            
            self.W = self.compute_weight_matrix()
            
            results = self.maximize(data)
            """
            for i in range(self.Nbins):
                for k in [f'X{j+1}' for j in range(self.Npc[i])]:
                    if np.abs(np.mean(results[k][i]) / np.std(results[k][i])) < 3 / np.sqrt(len(results[k][i])):
                        self.components[i].remove(k)
                        self.Ncomps -= 1
                        self.Npc[i] -= 1

            self.A_new = np.zeros((self.Ncross*self.Nbins, self.Ncomps))
            self.A_old = np.copy(self.A)

            c_index = 0
            for i in range(self.Nbins):
                A_ell = np.zeros((self.Ncross, self.Npc[i]+len(idx_fix)))
                for j, idx in enumerate(idx_fix):
                    if comp[idx] == 'cmb':
                        A_ell[:, j] = 1
                    else:
                        A_ell[:, j] = self.f_ij(l=i, key=comp[idx])#eval('self._'+comp[idx])(i)

                for j in range(self.Npc[i]):
                    A_ell[:, j+len(idx_fix)] = self.A[i*self.Ncross : (i+1)*self.Ncross, i*len(comp)+len(idx_fix)+int(self.components[i][j+len(idx_fix)][1:])-1]

                self.A_new[i*self.Ncross : (i+1)*self.Ncross, c_index : c_index+self.Npc[i]+len(idx_fix)] = A_ell
                c_index += self.Npc[i] + len(idx_fix)

            self.A = self.A_new            
            self.W = self.compute_weight_matrix()
            
            results = self.maximize(data)
            """
            

        if Azzoni:
            N = len(data)
            keys = ['cmb', 'A', 'As', 'rho', 'beta_d', 'T_d', 'beta_s', 'B_bb', 'y_bb', 'B_tt', 'y_tt', 'B_bt', 'y_bt', 'B_bsbs', 'y_bsbs', 'chi2r']
            Npars = len(keys) - 1
            Npars_ell = 4
            Npars_fixed = Npars - Npars_ell
            
            self.wigner = np.zeros((self.Nbins, self.Nbins, self.Nbins))
            for i, l in enumerate(self.leff):
                for j, l1 in enumerate(self.leff):
                    for k, l2 in enumerate(self.leff):
                        self.wigner[i, j, k] = (2*l1+1)*(2*l2+1)/(4*np.pi) * wigner_3j(l, l1, l2, 0, 0, 0)**2

            self._w2t = self._mbb_derivative('1/T', 2)
            def _w2t_uK(nu, l):
                S_nu = self._w2t(nu, self.beta_d[l], self.T_d[l], self.nu0_d)

                if np.array(nu).ndim == 2:
                    Ngrid = nu.shape[1]
                    weights = np.ones_like(nu)
                    bw = np.max(nu, axis=1) - np.min(nu, axis=1)
                    weights /= np.tile(bw, [Ngrid,1]).T
                    S_nu = np.trapezoid(S_nu * weights, nu)

                return S_nu * func.bandpass_unit_conversion(nu, input_unit='MJy/sr', output_unit='uK_CMB') / func.unit_conversion(self.nu0_d, input_unit='MJy/sr', output_unit='uK_CMB')
            self._w2t_uK = _w2t_uK

            self._w2bt = self._mbb_derivative(['beta', '1/T'], [1, 1])
            def _w2bt_uK(nu, l):
                S_nu = self._w2bt(nu, self.beta_d[l], self.T_d[l], self.nu0_d)

                if np.array(nu).ndim == 2:
                    Ngrid = nu.shape[1]
                    weights = np.ones_like(nu)
                    bw = np.max(nu, axis=1) - np.min(nu, axis=1)
                    weights /= np.tile(bw, [Ngrid,1]).T
                    S_nu = np.trapezoid(S_nu * weights, nu)

                return S_nu * func.bandpass_unit_conversion(nu, input_unit='MJy/sr', output_unit='uK_CMB') / func.unit_conversion(self.nu0_d, input_unit='MJy/sr', output_unit='uK_CMB')
            self._w2bt_uK = _w2bt_uK

            def _Aw2t(l):
                return 1/2 * (func.mbb_uK(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)*self._w2t_uK(self.nu_j, l) + self._w2t_uK(self.nu_i, l)*func.mbb_uK(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d))
            self._Aw2t = _Aw2t

            def _Aw2bt(l):
                return func.mbb_uK(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)*self._w2bt_uK(self.nu_j, l) + self._w2bt_uK(self.nu_i, l)*func.mbb_uK(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)
            self._Aw2bt = _Aw2bt

            pars = [{'value': 0, 'fixed': 0, 'limited': [1,1], 'limits': [-np.inf, np.inf]} for i in range(Npars_ell*self.Nbins + Npars_fixed)]
            for i in range(self.Nbins):
                pars[i] = pars[i] # cmb
                pars[self.Nbins + i]['limits'] = [0, np.inf] # A
                pars[2*self.Nbins + i]['limits'] = [0, np.inf] # As
                pars[3*self.Nbins + i]['limits'] = [-1, 1] # rho
            pars[4*self.Nbins]['limits'], pars[4*self.Nbins]['value'] = [1, 2], 1.48 # beta_d
            pars[4*self.Nbins + 1]['limits'], pars[4*self.Nbins + 1]['value'] = [15, 25], 19.6 # T_d
            pars[4*self.Nbins + 2]['limits'], pars[4*self.Nbins + 2]['value'] = [-4, -2], -3.1 # beta_s
            pars[4*self.Nbins + 3]['limits'] = [0, np.inf] # B_bb
            pars[4*self.Nbins + 4]['limits'], pars[4*self.Nbins + 4]['value'] = [-6, -2], -4 # y_bb
            pars[4*self.Nbins + 5]['limits'] = [0, np.inf] # B_tt
            pars[4*self.Nbins + 6]['limits'], pars[4*self.Nbins + 6]['value'] = [-6, -2], -4 # y_tt
            pars[4*self.Nbins + 7]['limits'] = [0, np.inf] # B_bt
            pars[4*self.Nbins + 8]['limits'], pars[4*self.Nbins + 8]['value'] = [-6, -2], -4 # y_bt
            pars[4*self.Nbins + 9]['limits'] = [0, np.inf] # B_bsbs
            pars[4*self.Nbins + 10]['limits'], pars[4*self.Nbins + 10]['value'] = [-6, -2], -4 # y_bsbs

            results = {k: 0 for k in keys}
            for i in range(Npars_ell):
                results[keys[i]] = np.zeros((self.Nbins, N))
            for i in range(Npars_fixed+1):
                results[keys[i+Npars_ell]] = np.zeros(N)

            L_inv = np.linalg.cholesky(self.N_inv)
            for k in trange(1, desc='Fits à la Azzoni...'):
                fa = {'y': np.concatenate(data[k].T), 'err': L_inv, 'model_func': self._Azzoni}
                self.m = mpfit(ftl.lkl_mpfit, parinfo=pars, functkw=fa, quiet=1)

                for i in range(Npars_ell):
                    results[keys[i]][:, k] = self.m.params[i*self.Nbins + np.arange(self.Nbins)]
                for i in range(Npars_fixed):
                    results[keys[i+Npars_ell]][k] = self.m.params[Npars_ell*self.Nbins + i]

                results['chi2r'][k] = self.m.fnorm / self.m.dof

        print('Done!')

        return results

    ############## Internal functions for model definition ##############
    ##############  Components are computed using beta(l)  ##############

    def _symbolic_derivative_mbb(self, variable, order):
        """
        Compute the symbolic derivative of a modified black-body
        """
        nu, beta, T, nu0 = sym.symbols('nu beta T nu_0')
        h, c, k = sym.symbols('h c k')

        x = h*nu / (k*T)
        x0 = h*nu0 / (k*T)

        B_nu = 2*h*nu**3/c**2 / (sym.exp(x) - 1) * 1e20
        B_nu0 = 2*h*nu0**3/c**2 / (sym.exp(x0) - 1) * 1e20
        I_nu = (nu/nu0)**beta * B_nu/B_nu0

        if type(variable) == str:
            variable = [variable]
            order = [order]

        for i in range(len(variable)):
            if variable[i] == 'beta':
                I_nu *= sym.log(nu/nu0)**order[i]
            elif variable[i] == 'T':
                I_nu = sym.diff(I_nu, T, order[i])
            elif variable[i] == '1/T':
                _T = sym.symbols('1/T')
                I_nu = sym.diff(I_nu.subs(T, 1/_T), _T, order[i]).subs(_T, 1/T)
            elif variable[i] == 'log(T)':
                logT = sym.symbols('log(T)')
                I_nu = sym.diff(I_nu.subs(T, sym.exp(logT)), logT, order[i]).subs(logT, sym.log(T))

        return I_nu

    def _mbb_derivative(self, variable, order):
        """
        Generate function for computing MBB derivative at given orders with respect to specified variables
        """
        nu, beta, T, nu0 = sym.symbols('nu beta T nu_0')
        h, c, k = sym.symbols('h c k')

        derivative = self._symbolic_derivative_mbb(variable, order).subs([(nu, nu*1e9), (nu0, nu0*1e9), (h, const.h.value), (c, const.c.value), (k, const.k_B.value)])

        return sym.lambdify((nu, beta, T, nu0), derivative, 'numpy')
    
    def _mbb_derivative_uK(self, nu, l, variable=None, order=0):
        """
        Compute MBB derivative at given orders with respect to specified variables in uK_CMB
        """
        if order == 0:
            return func.mbb_uK(nu, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)
        
        S_nu = self._mbb_derivative(variable, order)(nu, self.beta_d[l], self.T_d[l], self.nu0_d)

        if np.array(nu).ndim == 2:
            Ngrid = nu.shape[1]
            weights = np.ones_like(nu)
            bw = np.max(nu, axis=1) - np.min(nu, axis=1)
            weights /= np.tile(bw, [Ngrid,1]).T
            S_nu = np.trapezoid(S_nu * weights, nu)

        return S_nu * func.bandpass_unit_conversion(nu, input_unit='MJy/sr', output_unit='uK_CMB') / func.unit_conversion(self.nu0_d, input_unit='MJy/sr', output_unit='uK_CMB')

    def _pl_derivative_uK(self, nu, l, order):
        """
        Compute PL derivative at given orders with respect to specified variables in uK_CMB
        """
        S_nu = (nu/self.nu0_s)**self.beta_s[l] * np.log(nu/self.nu0_s)**order

        if np.array(nu).ndim < 2:
            return S_nu * func.bandpass_unit_conversion(nu, 'uK_RJ', 'uK_CMB') / func.unit_conversion(self.nu0_s, 'uK_RJ', 'uK_CMB')

        else:
            S_nu *= func.unit_conversion(nu, 'uK_RJ', 'MJy/sr') / func.unit_conversion(self.nu0_s, 'uK_RJ', 'MJy/sr')
            Ngrid = nu.shape[1]
            weights = np.ones_like(nu)
            bw = np.max(nu, axis=1) - np.min(nu, axis=1)
            weights /= np.tile(bw, [Ngrid,1]).T
            S_nu = np.trapezoid(S_nu * weights, nu)

            return S_nu * func.bandpass_unit_conversion(nu, input_unit='MJy/sr', output_unit='uK_CMB') / func.unit_conversion(self.nu0_s, input_unit='MJy/sr', output_unit='uK_CMB')

    def f_ij(self, l, key):
        """
        Compute the emissivity of the moment named 'key' in all cross-frequencies
        """
        vars = {'b': 'beta', 't': '1/T', 'bs': 'beta_s'}

        if key[0] == 'A':
            split = key.split('w')
            if len(split) == 1:
                if key == 'A':
                    var, order = [vars['b'], vars['b']], [0, 0]
                elif key == 'As':
                    var, order = [vars['bs'], vars['bs']], [0, 0]
                else:
                    var, order = [vars['b'], vars['bs']], [0, 0]

            else:
                if split[0] == 'A':
                    var, order = [vars['b']], [0]
                else:
                    var, order = [vars['bs']], [0]

                if split[1][1:] == 'bs':
                    var.append(vars['bs'])
                    order.append(int(split[1][0]))
                elif len(split[1][1:]) == 1:
                    var.append(vars[split[1][1:]])
                    order.append(int(split[1][0]))
                else:
                    var.append([vars['b'], vars['t']])
                    order.append([split[1][1:].count('b'), split[1][1:].count('t')])

        else:
            split = key.split('w')[1:]
            var, order = [], []
            for i in range(2):
                if split[i][1:] == 'bs':
                    var.append(vars['bs'])
                    order.append(int(split[i][0]))
                elif len(split[i][1:]) == 1:
                    var.append(vars[split[i][1:]])
                    order.append(int(split[i][0]))
                else:
                    var.append([vars['b'], vars['t']])
                    order.append([split[i][1:].count('b'), split[i][1:].count('t')])

        f_nu = np.zeros((2, 2, self.Ncross))
        for i in range(2):
            if var[i] == 'beta_s':
                f_nu[i] = 1 / np.prod(scipy.special.factorial(order[i])) * self._pl_derivative_uK(self.nu_i, l, order[i]), 1 / np.prod(scipy.special.factorial(order[i])) * self._pl_derivative_uK(self.nu_j, l, order[i])
            else:
                f_nu[i] = 1 / np.prod(scipy.special.factorial(order[i])) * self._mbb_derivative_uK(self.nu_i, l, var[i], order[i]), 1 / np.prod(scipy.special.factorial(order[i])) * self._mbb_derivative_uK(self.nu_j, l, var[i], order[i])

        f_ij = f_nu[0,0] * f_nu[1,1]
        if var[0] != var[1] or order[0] != order[1]:
            f_ij += f_nu[1,0] * f_nu[0,1]

        return f_ij

    
    def _A(self, l):
        return func.mbb_uK(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d) * func.mbb_uK(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)
    
    def _As(self, l):
        return func.PL_uK(self.nu_i, self.beta_s[l], nu0=self.nu0_s) * func.PL_uK(self.nu_j, self.beta_s[l], nu0=self.nu0_s)
    
    def _Asd(self, l):
        return func.mbb_uK(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)*func.PL_uK(self.nu_j, self.beta_s[l], nu0=self.nu0_s) + func.PL_uK(self.nu_i, self.beta_s[l], nu0=self.nu0_s)*func.mbb_uK(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)
    
    def _Aw1b(self, l):
        return func.mbb_uK(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)*func.dust_o1b(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d) + func.dust_o1b(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)*func.mbb_uK(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)
    
    def _Aw1t(self, l):
        return func.mbb_uK(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)*func.dust_o1t(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d) + func.dust_o1t(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)*func.mbb_uK(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)
    
    def _w1bw1b(self, l):
        return func.dust_o1b(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d) * func.dust_o1b(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)
    
    def _w1tw1t(self, l):
        return func.dust_o1t(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d) * func.dust_o1t(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)
    
    def _w1bw1t(self, l):
        return func.dust_o1b(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)*func.dust_o1t(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d) + func.dust_o1t(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)*func.dust_o1b(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)
    
    def _Asw1bs(self, l):
        return func.PL_uK(self.nu_i, self.beta_s[l], nu0=self.nu0_s)*func.sync_o1b(self.nu_j, self.beta_s[l], nu0=self.nu0_s) + func.sync_o1b(self.nu_i, self.beta_s[l], nu0=self.nu0_s)*func.PL_uK(self.nu_j, self.beta_s[l], nu0=self.nu0_s)
    
    def _w1bsw1bs(self, l):
        return func.sync_o1b(self.nu_i, self.beta_s[l], nu0=self.nu0_s) * func.sync_o1b(self.nu_j, self.beta_s[l], nu0=self.nu0_s)
    
    def _Asw1b(self, l):
        return func.PL_uK(self.nu_i, self.beta_s[l], nu0=self.nu0_s)*func.dust_o1b(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d) + func.dust_o1b(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)*func.PL_uK(self.nu_j, self.beta_s[l], nu0=self.nu0_s)
    
    def _Asw1t(self, l):
        return func.PL_uK(self.nu_i, self.beta_s[l], nu0=self.nu0_s)*func.dust_o1t(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d) + func.dust_o1t(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)*func.PL_uK(self.nu_j, self.beta_s[l], nu0=self.nu0_s)
    
    def _Aw1bs(self, l):
        return func.mbb_uK(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)*func.sync_o1b(self.nu_j, self.beta_s[l], nu0=self.nu0_s) + func.sync_o1b(self.nu_i, self.beta_s[l], nu0=self.nu0_s)*func.mbb_uK(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)
    
    def _w1bw1bs(self, l):
        return func.dust_o1b(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)*func.sync_o1b(self.nu_j, self.beta_s[l], nu0=self.nu0_s) + func.sync_o1b(self.nu_i, self.beta_s[l], nu0=self.nu0_s)*func.dust_o1b(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)
    
    def _w1tw1bs(self, l):
        return func.dust_o1t(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)*func.sync_o1b(self.nu_j, self.beta_s[l], nu0=self.nu0_s) + func.sync_o1b(self.nu_i, self.beta_s[l], nu0=self.nu0_s)*func.dust_o1t(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)

    def _Aw2b(self, l):
        return 1/2 * (func.mbb_uK(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)*func.dust_o2b(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d) + func.dust_o2b(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)*func.mbb_uK(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d))

    def _w1bw2b(self, l):
        return 1/2 * (func.dust_o1b(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)*func.dust_o2b(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d) + func.dust_o2b(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d)*func.dust_o1b(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d))

    def _w2bw2b(self, l):
        return 1/4 * (func.dust_o2b(self.nu_i, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d) * func.dust_o2b(self.nu_j, self.beta_d[l], 1/self.T_d[l], nu0=self.nu0_d))

    def _cmb(self, l):
        return np.ones(self.Ncross)
    
    def _r(self, l):
        return self.Dl_tens[l] * np.ones(self.Ncross)
    
    def _A_w1bsw1bs(self, l):
        return (self.leff[l] / 10)**self.gamma_w1bsw1bs * self._w1bsw1bs(l)

    def _w1_w1bsw1bs(self, l):
        return (self.leff[l] / 10)**self.gamma_w1bsw1bs * np.log(self.leff[l] / 10) * self._w1bsw1bs(l)

    def _A_w1bw1bs(self, l):
        return (self.leff[l] / 10)**self.gamma_w1bw1bs * self._w1bw1bs(l)

    def _w1_w1bw1bs(self, l):
        return (self.leff[l] / 10)**self.gamma_w1bw1bs * np.log(self.leff[l] / 10) * self._w1bw1bs(l)

    def _A_w1tw1bs(self, l):
        return (self.leff[l] / 10)**self.gamma_w1tw1bs * self._w1tw1bs(l)

    def _w1_w1tw1bs(self, l):
        return (self.leff[l] / 10)**self.gamma_w1tw1bs * np.log(self.leff[l] / 10) * self._w1tw1bs(l)

    def _A_Asw1bs(self, l):
        return (self.leff[l] / 10)**self.gamma_Asw1bs * self._Asw1bs(l)

    def _w1_Asw1bs(self, l):
        return (self.leff[l] / 10)**self.gamma_Asw1bs * np.log(self.leff[l] / 10) * self._Asw1bs(l)

    def _Azzoni(self, p):
        cmb = p[np.arange(self.Nbins)]
        A = p[self.Nbins + np.arange(self.Nbins)]
        As = p[2*self.Nbins + np.arange(self.Nbins)]
        rho = p[3*self.Nbins + np.arange(self.Nbins)]
        beta_d = p[4*self.Nbins]
        T_d = p[4*self.Nbins + 1]
        beta_s = p[4*self.Nbins + 2]
        B_bb = p[4*self.Nbins + 3]
        y_bb = p[4*self.Nbins + 4]
        B_tt = p[4*self.Nbins + 5]
        y_tt = p[4*self.Nbins + 6]
        B_bt = p[4*self.Nbins + 7]
        y_bt = p[4*self.Nbins + 8]
        B_bsbs = p[4*self.Nbins + 9]
        y_bsbs = p[4*self.Nbins + 10]

        self.beta_d[:] = beta_d
        self.T_d[:] = T_d
        self.beta_s[:] = beta_s

        Cl_AA = A / self.leff / (self.leff+1) * 2*np.pi
        Cl_AsAs = As / self.leff / (self.leff+1) * 2*np.pi
        
        Cl_bb = B_bb * (self.leff/10)**y_bb
        Cl_tt = B_tt * (self.leff/10)**y_tt
        Cl_bt = B_bt * (self.leff/10)**y_bt
        Cl_bsbs = B_bsbs * (self.leff/10)**y_bsbs

        sigma2_b = np.sum((2*self.leff+1)/(4*np.pi) * Cl_bb)
        sigma2_t = np.sum((2*self.leff+1)/(4*np.pi) * Cl_tt)
        sigma2_bs = np.sum((2*self.leff+1)/(4*np.pi) * Cl_bsbs)

        model = np.zeros((self.Ncross, self.Nbins))
        for l in range(self.Nbins):
            d_o0 = A[l] * self._A(l)
            s_o0 = As[l] * self._As(l)
            ds_o0 = rho[l] * np.sqrt(A[l]*As[l]) * self._Asd(l)
            
            d_o1b = np.einsum('ij,i,j', self.wigner[l], Cl_AA, Cl_bb) * self._w1bw1b(l)
            d_o1t = np.einsum('ij,i,j', self.wigner[l], Cl_AA, Cl_tt) * self._w1tw1t(l)
            d_o1bt = np.einsum('ij,i,j', self.wigner[l], Cl_AA, Cl_bt) * self._w1bw1t(l)
            s_o1bs = np.einsum('ij,i,j', self.wigner[l], Cl_AsAs, Cl_bsbs) * self._w1bsw1bs(l)
    
            d_o2b = Cl_AA[l] * sigma2_b * self._Aw2b(l)
            d_o2t = Cl_AA[l] * sigma2_t * self._Aw2t(l)
            d_o2bt = Cl_AA[l] * np.sqrt(sigma2_b*sigma2_t) * self._Aw2bt(l)
            s_o2bs = Cl_AsAs[l] * sigma2_bs * self.f_ij(l, 'Asw2bs')

            model[:, l] = cmb[l] + d_o0 + s_o0 + ds_o0 + d_o1b + d_o1t + d_o1bt + s_o1bs + d_o2b + d_o2t + d_o2bt + s_o2bs

        return np.concatenate(model.T)