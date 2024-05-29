import numpy as np
from mpfit import mpfit
import mpfitlib as mpl
import scipy
import matplotlib.pyplot as plt 
import basicfunc as func
import mpi4py
from mpi4py import MPI
import math
from tqdm import tqdm

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

def fitmbb_PL_parallel(nucross,DL,Linv,p0,quiet=True):
    """
    Fit a mbb, pl and r on a DL
    :param: nucross, array of the cross-frequencies
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param Linv: inverse of the Cholesky matrix
    :param quiet: display output of the fit for debugging
    :return results: dictionnary containing A, beta, temp, A_s, beta_s, r and X2red for each (ell,n)
    """
    N,_,Nell=DL.shape

    #parallel:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    perrank = math.ceil(N/size)

    nparam = len(p0)
    paramiterl=np.zeros((Nell,N,nparam+1))
    chi2l=np.zeros((Nell,N))
    funcfit=mpl.Fitdscordre0
    for L in tqdm(range(Nell)):
        pl0 = np.append(p0,L)
        parinfopl = [{'value':pl0[i], 'fixed':0} for i in range(nparam-1)] #fg params
        parinfopl.append({'value':pl0[nparam-1], 'fixed':0}) #add r    
        parinfopl.append({'value':pl0[nparam],'fixed':1}) #and L 
        for n in range(rank*perrank, (rank+1)*perrank):
            fa = {'x':nucross, 'y':DL[n,:,L], 'err': Linv[L]}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'A_s': paramiterl[:,:,3], 'beta_s': paramiterl[:,:,4],'A_sd' : paramiterl[:,:,5], 'r':paramiterl[:,:,6], 'X2red': chi2l}
    return results

def fitmbb_PL_parallelvec(nucross,DL,Linv,p0,quiet=True):
    """
    Fit a mbb, pl and r on a DL
    :param: nucross, array of the cross-frequencies
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param Linv: inverse of the Cholesky matrix
    :param quiet: display output of the fit for debugging
    :return results: dictionnary containing A, beta, temp, A_s, beta_s, r and X2red for each (ell,n)
    """
    N,_,Nell=DL.shape

    #parallel:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    perrank = math.ceil(N/size)

    ncross=len(nucross)
    nnus = int((-1 + np.sqrt(ncross * 8 + 1)) / 2.)
    posauto = [int(nnus * i - i * (i + 1) / 2 + i) for i in range(nnus)]
    nu = nucross[posauto]

    freq_pairs = np.array([(i, j) for i in range(nnus) for j in range(i, nnus)])
    nu_i = nu[freq_pairs[:, 0]]
    nu_j = nu[freq_pairs[:, 1]]

    nparam = len(p0)
    paramiterl=np.zeros((Nell,N,nparam+1))
    chi2l=np.zeros((Nell,N))
    funcfit=mpl.Fitdscordre0_vectorize

    for L in tqdm(range(Nell)):
        pl0 = np.append(p0,L)
        parinfopl = [{'value':pl0[i], 'fixed':0} for i in range(nparam-1)] #fg params
        parinfopl.append({'value':pl0[nparam-1], 'fixed':0}) #add r    
        parinfopl.append({'value':pl0[nparam],'fixed':1}) #and L 
        for n in range(rank*perrank, (rank+1)*perrank):
            fa = {'x1':nu_i, 'x2':nu_j, 'y':DL[n,:,L], 'err': Linv[L]}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'A_s': paramiterl[:,:,3], 'beta_s': paramiterl[:,:,4],'A_sd' : paramiterl[:,:,5], 'r':paramiterl[:,:,6], 'X2red': chi2l}
    return results

def fito1_bT_PL_parallel(nucross,DL,Linv,resultsmbb_PL,quiet=True,fix=1,fixAw=0,fixcterm=0):
    """
    Fit using a first order moment expansion in both beta and T on a DL
    :param: nucross, array of the cross-frequencies
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param Linv: inverse of the Cholesky matrix
    :param resultsmbb: must be input mbb best fit in the format of fitmbb()
    :param quiet: display output of the fit for debugging
    :param fix: fix the 0th order parameters, 1=yes, 0=no.
    :return results: dictionnary containing A, beta, temp, Aw1b, w1bw1b, r and X2red for each (ell,n)
    """
    N,_,Nell=DL.shape

    #parallel:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    perrank = math.ceil(N/size)

    nparam=14
    paramiterl=np.zeros((Nell,N,nparam+1))
    chi2l=np.zeros((Nell,N))
    funcfit=mpl.FitdscbetaT
    for L in tqdm(range(Nell)):
        for n in range(rank*perrank, (rank+1)*perrank):
            # first o1 fit, dust fixed, mom free, r fixed
            parinfopl = [{'value':resultsmbb_PL['A'][L,n], 'fixed':fix},{'value':resultsmbb_PL['beta'][L,n], 'fixed':fix,'limited':[1,1],'limits':[0.5,3.]},{'value':resultsmbb_PL['temp'][L,n], 'fixed':fix, 'limited':[1,1],'limits':[10.,30.]},{'value':resultsmbb_PL['A_s'][L,n], 'fixed':fix},{'value':resultsmbb_PL['beta_s'][L,n], 'fixed':fix},{'value':resultsmbb_PL['A_sd'][L,n], 'fixed':fix},{'value':0, 'fixed':fixAw},{'value':0, 'fixed':0},{'value':0, 'fixed':fixAw},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':fixcterm},{'value':0, 'fixed':fixcterm}, {'value':0, 'fixed':0},{'value':L, 'fixed':1}] #dust params
            fa = {'x':nucross, 'y':DL[n,:,L], 'err': Linv[L]}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'A_s':paramiterl[:,:,3] , 'beta_s':paramiterl[:,:,4], 'A_sd':paramiterl[:,:,5], 'Aw1b' : paramiterl[:,:,6], 'w1bw1b' : paramiterl[:,:,7],'Aw1t' : paramiterl[:,:,8],'w1bw1t' : paramiterl[:,:,9],'w1tw1t' : paramiterl[:,:,10],'Asw1b' : paramiterl[:,:,11],'Asw1t' : paramiterl[:,:,12],'r' : paramiterl[:,:,13], 'X2red': chi2l}
    return results

def fito1_bT_PL_p0_parallel(nucross,DL,Linv,pl0,quiet=True,fix=1,fixAw=0,fixcterm=1):
    """
    Fit using a first order moment expansion in both beta and T on a DL
    :param: nucross, array of the cross-frequencies
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param Linv: inverse of the Cholesky matrix
    :param resultsmbb: must be input mbb best fit in the format of fitmbb()
    :param quiet: display output of the fit for debugging
    :param fix: fix the 0th order parameters, 1=yes, 0=no.
    :return results: dictionnary containing A, beta, temp, Aw1b, w1bw1b, r and X2red for each (ell,n)
    """
    N,_,Nell=DL.shape
    #parallel:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    perrank = math.ceil(N/size)

    nparam=len(pl0)+1
    paramiterl=np.zeros((Nell,N,nparam))
    chi2l=np.zeros((Nell,N))
    funcfit=mpl.FitdscbetaT
    parinfopl0 = [{'value':pl0[i], 'fixed':0} for i in range(nparam-1)] #fg params
    for L in tqdm(range(Nell)):
        parinfopl=parinfopl0.copy()
        parinfopl.append({'value':L,'fixed':1}) #and L 
        for n in range(rank*perrank, (rank+1)*perrank):
            # first o1 fit, dust fixed, mom free, r fixed
            fa = {'x':nucross, 'y':DL[n,:,L], 'err': Linv[L]}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'A_s':paramiterl[:,:,3] , 'beta_s':paramiterl[:,:,4], 'A_sd':paramiterl[:,:,5], 'Aw1b' : paramiterl[:,:,6], 'w1bw1b' : paramiterl[:,:,7],'Aw1t' : paramiterl[:,:,8],'w1bw1t' : paramiterl[:,:,9],'w1tw1t' : paramiterl[:,:,10],'Asw1b' : paramiterl[:,:,11],'Asw1t' : paramiterl[:,:,12],'r' : paramiterl[:,:,13], 'X2red': chi2l}
    return results

def fito1_bT_PL_parallelvec(nucross,DL,Linv,pl0,quiet=True,fix=1,fixAw=0,fixcterm=0):
    """
    Fit using a first order moment expansion in both beta and T on a DL
    :param: nucross, array of the cross-frequencies
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param Linv: inverse of the Cholesky matrix
    :param resultsmbb: must be input mbb best fit in the format of fitmbb()
    :param quiet: display output of the fit for debugging
    :param fix: fix the 0th order parameters, 1=yes, 0=no.
    :return results: dictionnary containing A, beta, temp, Aw1b, w1bw1b, r and X2red for each (ell,n)
    """
    N,_,Nell=DL.shape
    nparam=14

    #parallel:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    perrank = math.ceil(N/size)

    ncross=len(nucross)
    nnus = int((-1 + np.sqrt(ncross * 8 + 1)) / 2.)
    posauto = [int(nnus * i - i * (i + 1) / 2 + i) for i in range(nnus)]
    nu = nucross[posauto]

    freq_pairs = np.array([(i, j) for i in range(nnus) for j in range(i, nnus)])
    nu_i = nu[freq_pairs[:, 0]]
    nu_j = nu[freq_pairs[:, 1]]

    parinfopl0 = [{'value':pl0[i], 'fixed':0} for i in range(nparam-1)] #fg params
    paramiterl=np.zeros((Nell,N,nparam+1))
    chi2l=np.zeros((Nell,N))
    funcfit=mpl.FitdscbetaT_vectorize
    for L in tqdm(range(Nell)):
        parinfopl=parinfopl0.copy()
        parinfopl.append({'value':L,'fixed':1}) #and L 
        for n in range(rank*perrank, (rank+1)*perrank):
            # first o1 fit, dust fixed, mom free, r fixed
            #parinfopl = [{'value':resultsmbb_PL['A'][L,n], 'fixed':fix},{'value':resultsmbb_PL['beta'][L,n],'fixed':fix,'limited':[1,1],'limits':[0.5,3.]},{'value':resultsmbb_PL['temp'][L,n], 'fixed':fix,'limited':[1,1],'limits':[10.,30.]},{'value':resultsmbb_PL['A_s'][L,n], 'fixed':fix},{'value':resultsmbb_PL['beta_s'][L,n], 'fixed':fix},{'value':resultsmbb_PL['A_sd'][L,n], 'fixed':fix},{'value':0, 'fixed':fixAw},{'value':0, 'fixed':0},{'value':0, 'fixed':fixAw},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':fixcterm},{'value':0, 'fixed':fixcterm}, {'value':0, 'fixed':0},{'value':L, 'fixed':1}] #dust params
            fa = {'x1':nu_i, 'x2':nu_j, 'y':DL[n,:,L], 'err': Linv[L]}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'A_s':paramiterl[:,:,3] , 'beta_s':paramiterl[:,:,4], 'A_sd':paramiterl[:,:,5], 'Aw1b' : paramiterl[:,:,6], 'w1bw1b' : paramiterl[:,:,7],'Aw1t' : paramiterl[:,:,8],'w1bw1t' : paramiterl[:,:,9],'w1tw1t' : paramiterl[:,:,10],'Asw1b' : paramiterl[:,:,11],'Asw1t' : paramiterl[:,:,12],'r' : paramiterl[:,:,13], 'X2red': chi2l}
    return results

def fito1_bT_moms_full_parallel(nucross,DL,Linv,resultsmbb_PL,quiet=True,fix=1):
    """
    Fit using a first order moment expansion in both beta and T on a DL
    :param: nucross, array of the cross-frequencies
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param Linv: inverse of the Cholesky matrix
    :param quiet: display output of the fit for debugging
    :param resultsmbb: must be input mbb best fit in the format of fitmbb()
    :return results: dictionnary containing A, beta, temp, Aw1b, w1bw1b, r and X2red for each (ell,n)
    """
    N,_,Nell=DL.shape

    #parallel:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    perrank = math.ceil(N/size)

    nparam=19
    paramiterl=np.zeros((Nell,N,nparam+1))
    chi2l=np.zeros((Nell,N))
    funcfit=mpl.FitdscbetaTbetas_full
    for L in tqdm(range(Nell)):
        for n in range(rank*perrank, (rank+1)*perrank):
            # first o1 fit, dust fixed, mom free, r fixed
            parinfopl = [{'value':resultsmbb_PL['A'][L,n], 'fixed':fix},{'value':resultsmbb_PL['beta'][L,n], 'fixed':fix},{'value':resultsmbb_PL['temp'][L,n], 'fixed':fix},{'value':resultsmbb_PL['A_s'][L,n], 'fixed':fix},{'value':resultsmbb_PL['beta_s'][L,n], 'fixed':fix},{'value':resultsmbb_PL['A_sd'][L,n], 'fixed':fix},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0}, {'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':L, 'fixed':1}] #dust params
            fa = {'x':nucross, 'y':DL[n,:,L], 'err': Linv[L]}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'A_s':paramiterl[:,:,3] , 'beta_s':paramiterl[:,:,4], 'A_sd':paramiterl[:,:,5], 'Aw1b' : paramiterl[:,:,6], 'w1bw1b' : paramiterl[:,:,7],'Aw1t' : paramiterl[:,:,8],'w1bw1t' : paramiterl[:,:,9],'w1tw1t' : paramiterl[:,:,10],'Asw1bs' : paramiterl[:,:,11],'w1bsw1bs' : paramiterl[:,:,12],'Asw1b' : paramiterl[:,:,13],'Asw1t' : paramiterl[:,:,14],'Adw1s' : paramiterl[:,:,15],'w1bw1s' : paramiterl[:,:,16],'w1sw1T' : paramiterl[:,:,17],'r' : paramiterl[:,:,18], 'X2red': chi2l}
    return results
