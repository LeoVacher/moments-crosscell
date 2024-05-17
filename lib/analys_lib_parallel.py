import numpy as np
from mpfit import mpfit
import mpfitlib as mpl
import scipy
import matplotlib.pyplot as plt 
import basicfunc as func

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

def fitmbb(nucross,DL,Linv,p0,quiet=True):
    """
    Fit a mbb and r on a DL
    :param: nucross, array of the cross-frequencies
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param Linv: inverse of the Cholesky matrix
    :param quiet: display output of the fit for debugging
    :return results: dictionnary containing A, beta, temp, r and X2red for each (ell,n)
    """
    N,_,Nell=DL.shape
    nparam = len(p0)
    paramiterl=np.zeros((Nell,N,nparam+1))
    chi2l=np.zeros((Nell,N))
    for L in range(0,Nell):
        print("%s%%"%(L*100/Nell))
        pl0 = np.append(p0,L)
        for n in range(N):
            fa = {'x':nucross, 'y':DL[n,:,L], 'err': Linv[L]}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'r' : paramiterl[:,:,3], 'X2red': chi2l}
    return results
