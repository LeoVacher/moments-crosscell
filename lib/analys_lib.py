import numpy as np
from mpfit import mpfit
import mpfitlib as mpl
import scipy
import matplotlib.pyplot as plt 
import basicfunc as func
from tqdm import tqdm
import mpi4py
from mpi4py import MPI
import plotlib as plib
import pymaster as nmt 

#contains all function for data analysis: matrix computations and moment fitting.

#GENERAL #######################################################################################################################

def getLinvdiag(DL,printdiag=False,offset=0):
    """
    Compute inverse of the covariance matrix used for the fit assuming it is block-diagonal in ell. 
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param print: if true, print the diagonal of cov.invcov to evaluate the quality of the inversion.
    :return Linv: Cholesky matrix in the shape (Nell,ncross,ncross)
    """
    _,_,Nell = DL.shape
    Linvdc = []
    DLtempo = np.swapaxes(DL,0,1)
    for L in range(Nell):
        cov = np.cov(DLtempo[:,:,L])
        invcov = np.linalg.inv(cov+offset*np.identity(len(cov)))
        if printdiag==True:
            print(np.diag(np.dot(cov,invcov)))
        Linvdc.append(np.linalg.cholesky(invcov))
    Linvdc = np.array(Linvdc)
    return Linvdc

def getLinv_all_ell(DL,printdiag=False,offset=0,Ncrdiag=0):
    """
    Compute inverse of the covariance matrix used for the fit assuming it is block-diagonal in ell. 
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param print: if true, print the diagonal of cov.invcov to evaluate the quality of the inversion.
    :return Linv: Cholesky matrix in the shape (Nellxncross,Nellxncross)
    """
    N,Ncross,Nell = DL.shape
    DLswap = np.swapaxes(DL,1,2)
    DLflat = np.zeros([Nell*Ncross,N])
    for i in range(N):
        DLflat[:,i] = DLswap[i,:,:].flatten()
    id = []
    for L in range(Nell):
        id.append(np.ones((Ncross,Ncross)))
    ident = np.zeros((Nell*Ncross,Nell*Ncross))
    for i in range(Nell):
        ident[i*Ncross:Ncross*(i+1),i*Ncross:Ncross*(i+1)]=id[i]
    for j in range(Ncrdiag):
           ident[j*Ncross:Ncross*(j+1),(j+1)*Ncross:Ncross*(j+2)]=id[i]
           ident[(j+1)*Ncross:Ncross*(j+2),j*Ncross:Ncross*(j+1)]=id[i]
    covtot = np.cov(DLflat)
    covtot = covtot*ident
    invcovtot = np.linalg.inv(covtot)
    if printdiag ==True:
        print(np.diag(np.dot(invcovtot,covtot)))
    Linvdc = np.linalg.cholesky(invcovtot)
    return Linvdc

# FIT FUNCTIONS ##################################################################################################################

def fit_mom(kw,nucross,DL,Linv,p0,quiet=True,parallel=False,nside = 64, Nlbin = 10,fix=1,all_ell=False,kwsave=""):
    """
    Fit using a first order moment expansion in both beta and T on a DL
    :param: kw, should be a string of the form 'X_Y' where X={d,s,ds} for dust,syncrotron or dust and syncrotron, and Y={o0,o1bt,o1bts} for order 0, first order in beta and T or first order in beta, T, betas
    :param: nucross, array of the cross-frequencies
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param Linv: inverse of the Cholesky matrix
    :param quiet: display output of the fit for debugging
    :param: parallel, if true use mpi to parallelise the computation on number of simulations.
    :return results: dictionnary containing A, beta, temp, Aw1b, w1bw1b, r and X2red for each (ell,n)
    """
    N,_,Nell=DL.shape
    nparam = len(p0)

    # get cmb spectra:

    DL_lensbin, DL_tens= mpl.getDL_cmb(nside=nside,Nlbin=Nlbin)

    #get frequencies:
    ncross=len(nucross)
    nnus = int((-1 + np.sqrt(ncross * 8 + 1)) / 2.)
    posauto = [int(nnus * i - i * (i + 1) / 2 + i) for i in range(nnus)]
    nu = nucross[posauto]
    freq_pairs = np.array([(i, j) for i in range(nnus) for j in range(i, nnus)])
    nu_i = nu[freq_pairs[:, 0]]
    nu_j = nu[freq_pairs[:, 1]]
    
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
    
        #select function to fit:
        funcfit= eval('mpl.func_'+kw)

        #set initial values:
        parinfopl =  [{'value':p0[i], 'fixed':0} for i in range(nparam)] #fg params
        parinfopl[0]= {'value':p0[0], 'fixed':0,'limited':[1,0],'limits':[0,np.inf]} #Ad
        parinfopl[1]= {'value':p0[1], 'fixed':fix,'limited':[1,1],'limits':[0.5,2]} #betad
        parinfopl[2]= {'value':p0[2], 'fixed':fix,'limited':[1,1],'limits':[3,100]} #Td
        parinfopl[3]= {'value':p0[3], 'fixed':0,'limited':[1,0],'limits':[0,np.inf]} #As
        parinfopl[4]= {'value':p0[4], 'fixed':fix,'limited':[1,1],'limits':[-5,-2]} #betas    
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
                fa = {'x1':nu_i, 'x2':nu_j, 'y':DL[n,:,L], 'err': Linv[L],'ell':L, 'DL_lensbin': DL_lensbin, 'DL_tens': DL_tens}
                m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
                paramiterl[L,n]= m.params
                chi2l[L,n]=m.fnorm/m.dof            
        
        #return result dictionnary:

        if kw=='ds_o0':
            results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'A_s': paramiterl[:,:,3], 'beta_s': paramiterl[:,:,4],'A_sd' : paramiterl[:,:,5], 'r':paramiterl[:,:,6], 'X2red': chi2l}
        elif kw=='ds_o1bt' or kw=='ds_o1bt_altnorm':
            results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'A_s':paramiterl[:,:,3] , 'beta_s':paramiterl[:,:,4], 'A_sd':paramiterl[:,:,5], 'Aw1b' : paramiterl[:,:,6], 'w1bw1b' : paramiterl[:,:,7],'Aw1t' : paramiterl[:,:,8],'w1bw1t' : paramiterl[:,:,9],'w1tw1t' : paramiterl[:,:,10],'Asw1b' : paramiterl[:,:,11],'Asw1t' : paramiterl[:,:,12],'r' : paramiterl[:,:,13], 'X2red': chi2l}
        elif kw=='ds_o1bts':
            results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'A_s':paramiterl[:,:,3] , 'beta_s':paramiterl[:,:,4], 'A_sd':paramiterl[:,:,5], 'Aw1b' : paramiterl[:,:,6], 'w1bw1b' : paramiterl[:,:,7],'Aw1t' : paramiterl[:,:,8],'w1bw1t' : paramiterl[:,:,9],'w1tw1t' : paramiterl[:,:,10],'Asw1bs' : paramiterl[:,:,11],'w1bsw1bs' : paramiterl[:,:,12],'Asw1b' : paramiterl[:,:,13],'Asw1t' : paramiterl[:,:,14],'Adw1s' : paramiterl[:,:,15],'w1bw1s' : paramiterl[:,:,16],'w1sw1T' : paramiterl[:,:,17],'r' : paramiterl[:,:,18], 'X2red': chi2l}
        else:
            print('unexisting keyword')
    if all_ell==True:
        funcfit= eval('mpl.func_'+kw+'_all_ell')

        #set initial values:
        parinfopl =  []
        [parinfopl.append({'value':p0[0], 'fixed':0,'limited':[1,0],'limits':[0,np.inf]}) for i in range(Nell)] #A_d
        [parinfopl.append({'value':p0[3], 'fixed':0,'limited':[1,0],'limits':[0,np.inf]}) for i in range(Nell)] #A_s
        [parinfopl.append({'value':p0[5], 'fixed':0,'limited':[0,0],'limits':[-np.inf,np.inf]}) for i in range(Nell)] #A_sd
        parinfopl.append({'value':p0[1], 'fixed':0,'limited':[1,1],'limits':[0.5,2]}) #betad
        parinfopl.append({'value':p0[2], 'fixed':0,'limited':[1,1],'limits':[3,100]}) #Td
        parinfopl.append({'value':p0[4], 'fixed':0,'limited':[1,1],'limits':[-5,-2]}) #betas    
        parinfopl.append({'value':p0[5], 'fixed':0}) #r 
        if kw=='ds_o1bt':
            [parinfopl.append({'value':0,'fixed':0}) for i in range(7)] #moments 
        elif kw=='ds_o1bts':
            [parinfopl.append({'value':0,'fixed':0}) for i in range(10)] #moments 
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
            fa = {'x1':nu_i, 'x2':nu_j, 'y':DLdcflat[n], 'err': Linv, 'DL_lensbin': DL_lensbin, 'DL_tens': DL_tens,'Nell':Nell}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiter[n]= m.params
            chi2[n]=m.fnorm/m.dof            
        
        #return result dictionnary:

        if kw=='ds_o0':
            results={'A' : np.swapaxes(paramiter[:,:Nell],0,1), 'beta' : paramiter[:,3*Nell], 'temp' : paramiter[:,3*Nell+1], 'A_s': np.swapaxes(paramiter[:,Nell:2*Nell],0,1), 'beta_s': paramiter[:,3*Nell+2],'A_sd' : np.swapaxes(paramiter[:,2*Nell:3*Nell],0,1), 'r':paramiter[:,3*Nell+3], 'X2red': chi2}
        elif kw=='ds_o1bt':
            results={'A' : np.swapaxes(paramiter[:,:Nell],0,1), 'beta' : paramiter[:,3*Nell], 'temp' : paramiter[:,3*Nell+1], 'A_s':np.swapaxes(paramiter[:,Nell:2*Nell],0,1) , 'beta_s':paramiter[:,3*Nell+2], 'A_sd':np.swapaxes(paramiter[:,2*Nell:3*Nell],0,1), 'Aw1b' : paramiter[:,3*Nell+4], 'w1bw1b' : paramiter[:,3*Nell+5],'Aw1t' : paramiter[:,3*Nell+6],'w1bw1t' : paramiter[:,3*Nell+7],'w1tw1t' : paramiter[:,3*Nell+8],'Asw1b' : paramiter[:,3*Nell+9],'Asw1t' : paramiter[:,3*Nell+10],'r' : paramiter[:,3*Nell+3], 'X2red': chi2}
        elif kw=='ds_o1bts':
            results={'A' : np.swapaxes(paramiter[:,:Nell],0,1), 'beta' : paramiter[:,3*Nell], 'temp' : paramiter[:,3*Nell+1], 'A_s':np.swapaxes(paramiter[:,Nell:2*Nell],0,1) , 'beta_s':paramiter[:,3*Nell+2], 'A_sd':np.swapaxes(paramiter[:,2*Nell:3*Nell],0,1), 'Aw1b' : paramiter[:,3*Nell+4], 'w1bw1b' : paramiter[:,3*Nell+4],'Aw1t' : paramiter[:,3*Nell+6],'w1bw1t' : paramiter[:,3*Nell+7],'w1tw1t' : paramiter[:,3*Nell+8],'Asw1bs' : paramiter[:,3*Nell+9],'w1bsw1bs' : paramiter[:,3*Nell+10],'Asw1b' : paramiter[:,3*Nell+11],'Asw1t' : paramiter[:,3*Nell+12],'Adw1s' : paramiter[:,3*Nell+13],'w1bw1s' : paramiter[:,3*Nell+14],'w1sw1T' : paramiter[:,3*Nell+15],'r' : paramiter[:,3*Nell+3], 'X2red': chi2}
        else:
            print('unexisting keyword')

    #save and plot results

    kw=kw+'_fix%s'%fix
    if all_ell==True:
        kw=kw+"_all_ell"
    np.save('./Best-fits/results_%s_%s.npy'%(kwsave,kw),results)
    b = nmt.bins.NmtBin(nside=nside,lmax=nside*3-1,nlb=Nlbin)
    l = b.get_effective_ells()
    plib.plotrespdf(l[:Nell],[results],['%s-%s'%(kwsave,kw)],['darkorange'])
    if all_ell==True:
        plib.plotr_hist(results,color='darkorange',save=True,kwsave='%s-%s'%(kwsave,kw))
    else:
        plib.plotr_gaussproduct(results,Nmax=Nell,debug=False,color='darkorange',save=True,kwsave='%s-%s'%(kwsave,kw))
    return results

