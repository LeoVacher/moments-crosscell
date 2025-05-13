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
import analytical_mom_lib as anmomlib

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

def fit_mom(kw,nucross,DL,Linv,p0,quiet=True,parallel=False,nside = 64, Nlbin = 10,fix=1,all_ell=False,adaptative=False,kwsave="",plotres=False,mompl=False,iterate=False,nu0d=353.,nu0s=23.,fixr=0, mask=None):
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
    :param plotres: if true, plot and save the results in pdf format.
    :param mompl: only for allell case, fit moments as power-laws in ell.
    :param iterate: if True, iterate the fit of the moments to estimate the best pivot value.
    :param fixr: if 1, fix the tensor to scalar ratio (r) to zero and does not fit for it.
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
    if adaptative:
        kwf = kwf+'_adaptive'
    if iterate:
        kwf = kwf+'_iterate'

    #create folder for parallel    
    if parallel == True:
        pathlib.Path('./best_fits/results_%s_%s.npy'%(kwsave,kwf)).mkdir(parents=True, exist_ok=True)

    # get cmb spectra:
    DL_lensbin, DL_tens= ftl.getDL_cmb(nside=nside,Nlbin=Nlbin)

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
            res0=np.load('./best_fits/results_%s_%s.npy'%(kwsave,kwf),allow_pickle=True).item()
            keys= res0.keys()
            for k in range(6,len(res0.keys())-2):
                for L in range(Nell):
                    fixmom=adaptafix(res0[list(keys)[k]][L])
                    parinfopl[L][k]= {'value':0, 'fixed':fixmom}

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
            else:
                while any(adaptafix(paramiterl[i, :, 6]) == 0 for i in range(Nell)):
                    for n in tqdm(range(Nmin,Nmax)):
                        for L in range(Nell):
                            parinfopl[L][1] = {'value': paramiterl[L,n,1] + paramiterl[L,n,6]/paramiterl[L,n,0] , 'fixed':1}
                            parinfopl[L][2] = {'value': paramiterl[L,n,2] + paramiterl[L,n,8]/paramiterl[L,n,0] , 'fixed':1}
                            if kw=='ds_o1bts':
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
            results={'A_d' : paramiterl[:,:,0], 'beta_d' : paramiterl[:,:,1], 'T_d' : 1/paramiterl[:,:,2], 'A_s':paramiterl[:,:,3] , 'beta_s':paramiterl[:,:,4], 'A_sd': paramiterl[:,:,5], 'Aw1b' : paramiterl[:,:,6], 'w1bw1b' : paramiterl[:,:,7],'Aw1t' : paramiterl[:,:,8],'w1bw1t' : paramiterl[:,:,9],'w1tw1t' : paramiterl[:,:,10],'Asw1bs' : paramiterl[:,:,11],'w1bsw1bs' : paramiterl[:,:,12],'Asw1b' : paramiterl[:,:,13],'Asw1t' : paramiterl[:,:,14],'Adw1s' : paramiterl[:,:,15],'w1bw1s' : paramiterl[:,:,16],'w1sw1T' : paramiterl[:,:,17],'r' : paramiterl[:,:,18], 'X2red': chi2l}
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
                results_mom= {'Aw1b' : paramiter[:,3*Nell+4], 'w1bw1b' : paramiter[:,3*Nell+4],'Aw1t' : paramiter[:,3*Nell+6],'w1bw1t' : paramiter[:,3*Nell+7],'w1tw1t' : paramiter[:,3*Nell+8],'Asw1bs' : paramiter[:,3*Nell+9],'w1bsw1bs' : paramiter[:,3*Nell+10],'Asw1b' : paramiter[:,3*Nell+11],'Asw1t' : paramiter[:,3*Nell+12],'Adw1s' : paramiter[:,3*Nell+13],'w1bw1s' : paramiter[:,3*Nell+14],'w1sw1T' : paramiter[:,3*Nell+15],'r' : paramiter[:,3*Nell+3], 'X2red': chi2}
                results_mom_pl = {}
            results = {**results_o0,**results_mom,**results_mom_pl}
        else:
            raise ValueError('unexisting keyword')

    #save and plot results
    
    if parallel:
        np.save('best_fits/results_%s_%s_p0/res%s.npy'%(kwsave,kwf,rank))    
    else:
        np.save('./best_fits/results_%s_%s.npy'%(kwsave,kwf),results)
        
        if plotres:
            dusttype,synctype = kwsave[1], kwsave[3]
            betabar = np.mean(results['beta_d'])
            tempbar = np.mean(results['T_d'])
            betasbar = np.mean(results['beta_s'])
            fsky = int(np.mean(mask)**2)
            try:
                mom_an = np.load('./analytical_mom/analytical_mom_nside%s_fsky%s_scale10_Nlbin10_d%ss%s_%s%s%s.npy' % (nside, fsky, dusttype, synctype, betabar, tempbar, betasbar), allow_pickle=True).item()
            except:
                print('Computing theoretical expecations for the fitted quantities ...')
                mom_an = anmomlib.getmom(dusttype, synctype, betabar, tempbar, betasbar, mask, Nlbin=Nlbin, nside=nside,nu0d=nu0d,nu0s=nu0s)
                np.save('./analytical_mom/analytical_mom_nside%s_fsky%s_scale10_Nlbin10_d%ss%s_%s%s%s_%s%s.npy' % (nside, fsky, dusttype, synctype, betabar, tempbar, betasbar,nu0d,nu0s), mom_an)
            if all_ell:
                plot_contours=False
            else:
                plot_contours=True
            print('Plotting the results ...')
            plib.plotrespdf(l[:Nell],[results],['%s-%s'%(kwsave,kwf)],['darkorange'],mom_an,plot_contours=plot_contours,betadbar=betabar,tempbar=tempbar,betasbar=betasbar)
            if all_ell:
                plib.plotr_hist(results,color='darkorange',save=True,kwsave='%s%s'%(kwsave,kwf))
            else:
                plib.plotr_gaussproduct(results,Nmax=Nell,debug=False,color='darkorange',save=True,kwsave='%s-%s'%(kwsave,kwf))
    return results
