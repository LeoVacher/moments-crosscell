import numpy as np
from mpfit import mpfit
import mpfitlib as mpl
import scipy
import matplotlib.pyplot as plt 
import basicfunc as func
from tqdm import tqdm
import mpi4py
from mpi4py import MPI

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

# FIT FUNCTIONS ##################################################################################################################

def fit_mom(kw,nucross,DL,Linv,p0,quiet=True,fix=1,parallel=False):
    """
    Fit using a first order moment expansion in both beta and T on a DL
    :param: kw, should be a string of the form 'X_Y' where X={d,s,ds} for dust,syncrotron or dust and syncrotron, and Y={o0,o1bt,o1bts} for order 0, first order in beta and T or first order in beta, T, betas
    :param: nucross, array of the cross-frequencies
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param Linv: inverse of the Cholesky matrix
    :param quiet: display output of the fit for debugging
    :param resultsmbb: must be input mbb best fit in the format of fitmbb()
    :param : parallel, if true use mpi to parallelise the computation on number of simulations.
    :return results: dictionnary containing A, beta, temp, Aw1b, w1bw1b, r and X2red for each (ell,n)
    """
    N,_,Nell=DL.shape
    nparam = len(p0)

    #get frequencies:
    ncross=len(nucross)
    nnus = int((-1 + np.sqrt(ncross * 8 + 1)) / 2.)
    posauto = [int(nnus * i - i * (i + 1) / 2 + i) for i in range(nnus)]
    nu = nucross[posauto]
    freq_pairs = np.array([(i, j) for i in range(nnus) for j in range(i, nnus)])
    nu_i = nu[freq_pairs[:, 0]]
    nu_j = nu[freq_pairs[:, 1]]

    #initialize parameters and chi2:
    paramiterl=np.zeros((Nell,N,nparam))
    chi2l=np.zeros((Nell,N))
    
    #select function to fit:
    funcfit= eval('mpl.func_'+kw)

    #set initial values:
    parinfopl = [{'value':p0[i], 'fixed':0} for i in range(nparam)] #fg params

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

    for L in tqdm(range(Nell)):
        for n in tqdm(range(Nmin,Nmax)):
            # first o1 fit, dust fixed, mom free, r fixed
            fa = {'x1':nu_i, 'x2':nu_j, 'y':DL[n,:,L], 'err': Linv[L],'ell':L}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    
    #return result dictionnary:

    if kw=='ds_o0':
        results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'A_s': paramiterl[:,:,3], 'beta_s': paramiterl[:,:,4],'A_sd' : paramiterl[:,:,5], 'r':paramiterl[:,:,6], 'X2red': chi2l}
    elif kw=='ds_o1bt':
        results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'A_s':paramiterl[:,:,3] , 'beta_s':paramiterl[:,:,4], 'A_sd':paramiterl[:,:,5], 'Aw1b' : paramiterl[:,:,6], 'w1bw1b' : paramiterl[:,:,7],'Aw1t' : paramiterl[:,:,8],'w1bw1t' : paramiterl[:,:,9],'w1tw1t' : paramiterl[:,:,10],'Asw1b' : paramiterl[:,:,11],'Asw1t' : paramiterl[:,:,12],'r' : paramiterl[:,:,13], 'X2red': chi2l}
    elif kw=='ds_o1bts':
        results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'A_s':paramiterl[:,:,3] , 'beta_s':paramiterl[:,:,4], 'A_sd':paramiterl[:,:,5], 'Aw1b' : paramiterl[:,:,6], 'w1bw1b' : paramiterl[:,:,7],'Aw1t' : paramiterl[:,:,8],'w1bw1t' : paramiterl[:,:,9],'w1tw1t' : paramiterl[:,:,10],'Asw1bs' : paramiterl[:,:,11],'w1bsw1bs' : paramiterl[:,:,12],'Asw1b' : paramiterl[:,:,13],'Asw1t' : paramiterl[:,:,14],'Adw1s' : paramiterl[:,:,15],'w1bw1s' : paramiterl[:,:,16],'w1sw1T' : paramiterl[:,:,17],'r' : paramiterl[:,:,18], 'X2red': chi2l}
    else:
        print('unexisting keyword')
    
    return results

