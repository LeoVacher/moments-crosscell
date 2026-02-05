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
    def __init__(self, freq, covmat, comp, beta_d, T_d, beta_s, nu0_d, nu0_s):
        """
        Initialize class by computing mixing matrix A and weight matrix W.
    
        Parameters
        ----------
        freq : array_like
            Array of frequencies at which the mixing matrix should be computed. Can be of shape (Nfreqs,) or (Nfreqs, Ngrid) depending on whether bandpass integration is taken into account.
        covmat : array_like
            Fiducial covariance matrix of dimension (Ncross*Nbins, Ncross*Nbins)
        comp : list
            List of components to be included in the mixing matrix for each bandpower. Each element of comp should be a list of components associated with the considered multipole bin.
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
    
        Returns
        -------
        None
        """
        self.Nfreqs = len(freq)
        self.Ncross = int(self.Nfreqs * (self.Nfreqs+1)/2)
        freq_pairs = np.array([(i, j) for i in range(self.Nfreqs) for j in range(i, self.Nfreqs)])
        self.nu_i = freq[freq_pairs[:, 0]]
        self.nu_j = freq[freq_pairs[:, 1]]
        
        self.Nbins = len(comp)
        self.components = [comp[i].copy() for i in range(self.Nbins)]
        self.Ncomps = 0
        for i in range(self.Nbins):
            self.Ncomps += len(comp[i])
        
        self.beta_d = beta_d * np.ones(self.Nbins)
        self.T_d = T_d * np.ones(self.Nbins)
        self.beta_s = beta_s * np.ones(self.Nbins)
        self.nu0_d, self.nu0_s = nu0_d, nu0_s
        
        self.N_inv = cvl.inverse_covmat(covmat, Ncross=self.Ncross, neglect_corbins=False)        
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
                    A_ell[:, j] = eval('self._'+c)(i)
    
            A[i*self.Ncross : (i+1)*self.Ncross, c_index : c_index+Nc] = A_ell
            c_index += Nc
    
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

            res = d - self.A @ s
            dof = len(d) - len(s)

            chi2r[k] = res.T @ self.N_inv @ res / dof

        keys = ['A', 'As', 'Asd', 'Aw1b', 'Aw1t', 'w1bw1b', 'w1tw1t', 'w1bw1t', 'Asw1bs', 'w1bsw1bs', 'Asw1b', 'Asw1t', 'Aw1bs', 'w1bw1bs', 'w1tw1bs', 'cmb']
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

        return results

    def run(self, data, n_iter=3, adaptative=True):
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

        Returns
        ----------
        results : dict
            Dictionnary of the estimated component amplitudes for the input simulations.
        """
        for i in trange(n_iter, desc='Iterations'):
            results = self.maximize(data)
            
            self.beta_d += np.mean(results['Aw1b'] / results['A'], axis=1)
            self.T_d = 1 / (1/self.T_d + np.mean(results['Aw1t'] / results['A'], axis=1))
            self.beta_s += np.mean(results['Asw1bs'] / results['As'], axis=1)

            self.A = self.compute_mixing_matrix()
            self.W = self.compute_weight_matrix()

        print('Fit simulations using updated pivot values...')
        if n_iter > 0:
            self.beta_d[:] = np.mean(self.beta_d)
            self.T_d[:] = np.mean(self.T_d)
            self.beta_s[:] = np.mean(self.beta_s)
            self.A = self.compute_mixing_matrix()
            self.W = self.compute_weight_matrix()

        results = self.maximize(data)

        if adaptative:
            print('Run adaptative fits...')
            dust_keys = ['Aw1b', 'Aw1t', 'w1bw1b', 'w1tw1t', 'w1bw1t', 'Asw1b', 'Asw1t']
            sync_keys = ['Asw1bs', 'w1bsw1bs', 'Aw1bs', 'w1bw1bs', 'w1tw1bs']

            for i in range(self.Nbins):
                if all(adaptafix(results[k][i]) == 1 for k in sync_keys):
                    for k in sync_keys:
                        self.components[i].remove(k)
                        self.Ncomps -= 1
                        
                    if all(adaptafix(results[k][i]) == 1 for k in dust_keys):
                        for k in dust_keys:
                            self.components[i].remove(k)
                            self.Ncomps -= 1
                
            self.A = self.compute_mixing_matrix()
            self.W = self.compute_weight_matrix()

            results = self.maximize(data)
            print('Done!')

        return results

    ############## Internal functions for model definition ##############
    ##############  Components are computed using beta(l)  ##############
    
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