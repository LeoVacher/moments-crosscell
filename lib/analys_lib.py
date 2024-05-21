import numpy as np
from mpfit import mpfit
import mpfitlib as mpl
import scipy
import matplotlib.pyplot as plt 
import basicfunc as func

#contains all function for data analysis: matrix computations, moment fitting and plot results.

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
    funcfit=mpl.Fitdcordre0
    for L in range(0,Nell):
        print("%s%%"%(L*100/Nell))
        pl0 = np.append(p0,L)
        parinfopl = [{'value':pl0[i], 'fixed':0} for i in range(nparam-1)] #dust params
        parinfopl.append({'value':pl0[nparam-1], 'fixed':0}) #add r    
        parinfopl.append({'value':pl0[nparam],'fixed':1}) #and L 
        for n in range(N):
            fa = {'x':nucross, 'y':DL[n,:,L], 'err': Linv[L]}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'r' : paramiterl[:,:,3], 'X2red': chi2l}
    return results


def fito1_b(nucross,DL,Linv,resultsmbb,quiet=True):
    """
    Fit using a first order moment expansion in beta on a DL
    :param: nucross, array of the cross-frequencies
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param Linv: inverse of the Cholesky matrix
    :param iter: desired iterations (not impelemented yet)
    :param resultsmbb: must be input mbb best fit in the format of fitmbb()
    :param quiet: display output of the fit for debugging
    :return results: dictionnary containing A, beta, temp, r and X2red for each (ell,n)
    """
    N,_,Nell=DL.shape
    nparam=6
    paramiterl=np.zeros((Nell,N,nparam+1))
    chi2l=np.zeros((Nell,N))
    funcfit=mpl.Fitdcordre1
    for L in range(0,Nell):
        print("%s%%"%(L*100/Nell))
        for n in range(N):
            # first o1 fit, dust fixed, mom free, r fixed
            parinfopl = [{'value':resultsmbb['A'][L,n], 'fixed':1},{'value':resultsmbb['beta'][L,n], 'fixed':1},{'value':resultsmbb['temp'][L,n], 'fixed':1},{'value':0, 'fixed':0},{'value':0, 'fixed':0}, {'value':0, 'fixed':0},{'value':L, 'fixed':1}] #dust params
            fa = {'x':nucross, 'y':DL[n,:,L], 'err': Linv[L]}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'Aw1b' : paramiterl[:,:,3], 'w1bw1b' : paramiterl[:,:,4], 'r' : paramiterl[:,:,5], 'X2red': chi2l}
    return results

def fito1_bT(nucross,DL,Linv,resultsmbb,quiet=True):
    """
    Fit using a first order moment expansion in both beta and T on a DL
    :param: nucross, array of the cross-frequencies
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param Linv: inverse of the Cholesky matrix
    :param resultsmbb: must be input mbb best fit in the format of fitmbb()
    :param quiet: display output of the fit for debugging
    :return results: dictionnary containing A, beta, temp, Aw1b, w1bw1b, r and X2red for each (ell,n)
    """
    N,_,Nell=DL.shape
    nparam=9
    paramiterl=np.zeros((Nell,N,nparam+1))
    chi2l=np.zeros((Nell,N))
    funcfit=mpl.FitdcbetaT
    for L in range(0,Nell):
        print("%s%%"%(L*100/Nell))
        for n in range(N):
            # first o1 fit, dust fixed, mom free, r fixed
            parinfopl = [{'value':resultsmbb['A'][L,n], 'fixed':1},{'value':resultsmbb['beta'][L,n], 'fixed':1},{'value':resultsmbb['temp'][L,n], 'fixed':1},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0}, {'value':0, 'fixed':0},{'value':L, 'fixed':1}] #dust params
            fa = {'x':nucross, 'y':DL[n,:,L], 'err': Linv[L]}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'Aw1b' : paramiterl[:,:,3], 'w1bw1b' : paramiterl[:,:,4],'Aw1t' : paramiterl[:,:,5],'w1bw1t' : paramiterl[:,:,6],'w1tw1t' : paramiterl[:,:,7], 'r' : paramiterl[:,:,8], 'X2red': chi2l}
    return results

def fito2_b(nucross,DL,Linv,resultsmbb,quiet=True):
    """
    Fit using a second order moment expansion in beta on a DL
    :param: nucross, array of the cross-frequencies
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param Linv: inverse of the Cholesky matrix
    :param resultsmbb: must be input mbb best fit in the format of fitmbb()
    :param quiet: display output of the fit for debugging
    :return results: dictionnary containing A, beta, temp, r and X2red for each (ell,n)
    """
    N,_,Nell=DL.shape
    nparam=9
    paramiterl=np.zeros((Nell,N,nparam+1))
    chi2l=np.zeros((Nell,N))
    funcfit=mpl.Fitdcordre2
    for L in range(0,Nell):
        print("%s%%"%(L*100/Nell))
        for n in range(N):
            parinfopl = [{'value':resultsmbb['A'][L,n], 'fixed':1},{'value':resultsmbb['beta'][L,n], 'fixed':1},{'value':resultsmbb['temp'][L,n], 'fixed':1},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0}, {'value':0, 'fixed':0},{'value':L, 'fixed':1}] #dust params
            fa = {'x':nucross, 'y':DL[n,:,L], 'err': Linv[L]}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'Aw1b' : paramiterl[:,:,3], 'w1bw1b' : paramiterl[:,:,4],'Aw2b' : paramiterl[:,:,5],'w1bw2b' : paramiterl[:,:,6],'w2bw2b' : paramiterl[:,:,7], 'r' : paramiterl[:,:,8], 'X2red': chi2l}
    return results

# add syncrotron

def fitmbb_PL(nucross,DL,Linv,p0,quiet=True):
    """
    Fit a mbb, pl and r on a DL
    :param: nucross, array of the cross-frequencies
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param Linv: inverse of the Cholesky matrix
    :param quiet: display output of the fit for debugging
    :return results: dictionnary containing A, beta, temp, A_s, beta_s, r and X2red for each (ell,n)
    """
    N,_,Nell=DL.shape
    nparam = len(p0)
    paramiterl=np.zeros((Nell,N,nparam+1))
    chi2l=np.zeros((Nell,N))
    funcfit=mpl.Fitdscordre0
    for L in range(0,Nell):
        print("%s%%"%(L*100/Nell))
        pl0 = np.append(p0,L)
        parinfopl = [{'value':pl0[i], 'fixed':0} for i in range(nparam-1)] #fg params
        parinfopl.append({'value':pl0[nparam-1], 'fixed':0}) #add r    
        parinfopl.append({'value':pl0[nparam],'fixed':1}) #and L 
        for n in range(N):
            fa = {'x':nucross, 'y':DL[n,:,L], 'err': Linv[L]}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'A_s': paramiterl[:,:,3], 'beta_s': paramiterl[:,:,4],'A_sd' : paramiterl[:,:,5], 'r':paramiterl[:,:,6], 'X2red': chi2l}
    return results

def fit_PL(nucross,DL,Linv,p0,quiet=True):
    """
    Fit a pl and r on a DL
    :param: nucross, array of the cross-frequencies
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param Linv: inverse of the Cholesky matrix
    :param quiet: display output of the fit for debugging
    :return results: dictionnary containing A, beta, temp, A_s, beta_s, r and X2red for each (ell,n)
    """
    N,_,Nell=DL.shape
    nparam = len(p0)
    paramiterl=np.zeros((Nell,N,nparam+1))
    chi2l=np.zeros((Nell,N))
    funcfit=mpl.Fitsc
    for L in range(0,Nell):
        print("%s%%"%(L*100/Nell))
        pl0 = np.append(p0,L)
        parinfopl = [{'value':pl0[i], 'fixed':0} for i in range(nparam-1)] #fg params
        parinfopl.append({'value':pl0[nparam-1], 'fixed':0}) #add r    
        parinfopl.append({'value':pl0[nparam],'fixed':1}) #and L 
        for n in range(N):
            fa = {'x':nucross, 'y':DL[n,:,L], 'err': Linv[L]}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A_s' : paramiterl[:,:,0], 'beta_s' : paramiterl[:,:,1], 'r' : paramiterl[:,:,2], 'X2red': chi2l}
    return results

def fito1_bs(nucross,DL,Linv,results_PL,quiet=True):
    """
    Fit a pl, first order moment expansion in beta_s and r on a DL
    :param: nucross, array of the cross-frequencies
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param Linv: inverse of the Cholesky matrix
    :param quiet: display output of the fit for debugging
    :return results: dictionnary containing A, beta, temp, A_s, beta_s, r and X2red for each (ell,n)
    """
    N,_,Nell=DL.shape
    nparam = 5
    paramiterl=np.zeros((Nell,N,nparam+1))
    chi2l=np.zeros((Nell,N))
    funcfit=mpl.Fitscordre1
    for L in range(0,Nell):
        print("%s%%"%(L*100/Nell))
        for n in range(N):
            parinfopl = [{'value':results_PL['A_s'][L,n], 'fixed':1},{'value':results_PL['beta_s'][L,n], 'fixed':1},{'value':0, 'fixed':0},{'value':0, 'fixed':0}, {'value':0, 'fixed':0},{'value':L, 'fixed':1}] #sync params
            fa = {'x':nucross, 'y':DL[n,:,L], 'err': Linv[L]}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A_s' : paramiterl[:,:,0], 'beta_s' : paramiterl[:,:,1],'Asw1bs' : paramiterl[:,:,2],'w1bsw1bs' : paramiterl[:,:,3], 'r' : paramiterl[:,:,4], 'X2red': chi2l}
    return results

def fito1_bT_PL(nucross,DL,Linv,resultsmbb_PL,quiet=True,fix=1,fixAw=0,fixcterm=1):
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
    paramiterl=np.zeros((Nell,N,nparam+1))
    chi2l=np.zeros((Nell,N))
    funcfit=mpl.FitdscbetaT
    for L in range(0,Nell):
        print("%s%%"%(L*100/Nell))
        for n in range(N):
            # first o1 fit, dust fixed, mom free, r fixed
            parinfopl = [{'value':resultsmbb_PL['A'][L,n], 'fixed':fix},{'value':resultsmbb_PL['beta'][L,n], 'fixed':fix},{'value':resultsmbb_PL['temp'][L,n], 'fixed':fix},{'value':resultsmbb_PL['A_s'][L,n], 'fixed':fix},{'value':resultsmbb_PL['beta_s'][L,n], 'fixed':fix},{'value':resultsmbb_PL['A_sd'][L,n], 'fixed':fix},{'value':0, 'fixed':fixAw},{'value':0, 'fixed':0},{'value':0, 'fixed':fixAw},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':fixcterm},{'value':0, 'fixed':fixcterm}, {'value':0, 'fixed':0},{'value':L, 'fixed':1}] #dust params
            fa = {'x':nucross, 'y':DL[n,:,L], 'err': Linv[L]}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'A_s':paramiterl[:,:,3] , 'beta_s':paramiterl[:,:,4], 'A_sd':paramiterl[:,:,5], 'Aw1b' : paramiterl[:,:,6], 'w1bw1b' : paramiterl[:,:,7],'Aw1t' : paramiterl[:,:,8],'w1bw1t' : paramiterl[:,:,9],'w1tw1t' : paramiterl[:,:,10],'Asw1b' : paramiterl[:,:,11],'Asw1t' : paramiterl[:,:,12],'r' : paramiterl[:,:,13], 'X2red': chi2l}
    return results

def fito2_bT_PL(nucross,DL,Linv,resultsmbb_PL,quiet=True,fix=1):
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
    nparam=15
    paramiterl=np.zeros((Nell,N,nparam+1))
    chi2l=np.zeros((Nell,N))
    funcfit=mpl.Fitdcbeta2T_PL
    for L in range(0,Nell):
        print("%s%%"%(L*100/Nell))
        for n in range(N):
            # first o1 fit, dust fixed, mom free, r fixed
            parinfopl = [{'value':resultsmbb_PL['A'][L,n], 'fixed':fix},{'value':resultsmbb_PL['beta'][L,n], 'fixed':fix},{'value':resultsmbb_PL['temp'][L,n], 'fixed':fix},{'value':resultsmbb_PL['A_s'][L,n], 'fixed':fix},{'value':resultsmbb_PL['beta_s'][L,n], 'fixed':fix},{'value':resultsmbb_PL['A_sd'][L,n], 'fixed':fix},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0}, {'value':0, 'fixed':0},{'value':L, 'fixed':1}] #dust params
            fa = {'x':nucross, 'y':DL[n,:,L], 'err': Linv[L]}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'A_s':paramiterl[:,:,3] , 'beta_s':paramiterl[:,:,4], 'A_sd':paramiterl[:,:,5], 'Aw1b' : paramiterl[:,:,6], 'w1bw1b' : paramiterl[:,:,7],'Aw1t' : paramiterl[:,:,8],'w1bw1t' : paramiterl[:,:,9],'w1tw1t' : paramiterl[:,:,10],'Aw2b' : paramiterl[:,:,11],'Aw1bw2b' : paramiterl[:,:,12],'Aw2bw2b' : paramiterl[:,:,13], 'r': paramiterl[:,:,14], 'X2red': chi2l}
    return results


def fito1_bT_moms_nocterm(nucross,DL,Linv,resultsmbb_PL,quiet=True,fix=1):
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
    nparam=14
    paramiterl=np.zeros((Nell,N,nparam+1))
    chi2l=np.zeros((Nell,N))
    funcfit=mpl.FitdscbetaTbetas_nocterm
    for L in range(0,Nell):
        print("%s%%"%(L*100/Nell))
        for n in range(N):
            # first o1 fit, dust fixed, mom free, r fixed
            parinfopl = [{'value':resultsmbb_PL['A'][L,n], 'fixed':fix},{'value':resultsmbb_PL['beta'][L,n], 'fixed':fix},{'value':resultsmbb_PL['temp'][L,n], 'fixed':fix},{'value':resultsmbb_PL['A_s'][L,n], 'fixed':fix},{'value':resultsmbb_PL['beta_s'][L,n], 'fixed':fix},{'value':resultsmbb_PL['A_sd'][L,n], 'fixed':fix},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0}, {'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':L, 'fixed':1}] #dust params
            fa = {'x':nucross, 'y':DL[n,:,L], 'err': Linv[L]}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'A_s':paramiterl[:,:,3] , 'beta_s':paramiterl[:,:,4], 'A_sd':paramiterl[:,:,5], 'Aw1b' : paramiterl[:,:,6], 'w1bw1b' : paramiterl[:,:,7],'Aw1t' : paramiterl[:,:,8],'w1bw1t' : paramiterl[:,:,9],'w1tw1t' : paramiterl[:,:,10],'Asw1bs' : paramiterl[:,:,11],'w1bsw1bs' : paramiterl[:,:,12],'r' : paramiterl[:,:,13], 'X2red': chi2l}
    return results

def fito1_bT_moms_full(nucross,DL,Linv,resultsmbb_PL,quiet=True,fix=1):
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
    nparam=19
    paramiterl=np.zeros((Nell,N,nparam+1))
    chi2l=np.zeros((Nell,N))
    funcfit=mpl.FitdscbetaTbetas_full
    for L in range(0,Nell):
        print("%s%%"%(L*100/Nell))
        for n in range(N):
            # first o1 fit, dust fixed, mom free, r fixed
            parinfopl = [{'value':resultsmbb_PL['A'][L,n], 'fixed':fix},{'value':resultsmbb_PL['beta'][L,n], 'fixed':fix},{'value':resultsmbb_PL['temp'][L,n], 'fixed':fix},{'value':resultsmbb_PL['A_s'][L,n], 'fixed':fix},{'value':resultsmbb_PL['beta_s'][L,n], 'fixed':fix},{'value':resultsmbb_PL['A_sd'][L,n], 'fixed':fix},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0}, {'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':L, 'fixed':1}] #dust params
            fa = {'x':nucross, 'y':DL[n,:,L], 'err': Linv[L]}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=quiet)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'A_s':paramiterl[:,:,3] , 'beta_s':paramiterl[:,:,4], 'A_sd':paramiterl[:,:,5], 'Aw1b' : paramiterl[:,:,6], 'w1bw1b' : paramiterl[:,:,7],'Aw1t' : paramiterl[:,:,8],'w1bw1t' : paramiterl[:,:,9],'w1tw1t' : paramiterl[:,:,10],'Asw1bs' : paramiterl[:,:,11],'w1bsw1bs' : paramiterl[:,:,12],'Asw1b' : paramiterl[:,:,13],'Asw1t' : paramiterl[:,:,14],'Adw1s' : paramiterl[:,:,15],'w1bw1s' : paramiterl[:,:,16],'w1sw1T' : paramiterl[:,:,17],'r' : paramiterl[:,:,18], 'X2red': chi2l}
    return results

#PLOT FUNCTIONS ##################################################################################################################

def plotr_gaussproduct(results,Nmin=0,Nmax=20,label='MBB',color='darkblue',debug=False,r=0,quiet=True,save=False,kwsave=''):
    """
    Fit a Gaussian curve for r(ell) in each bin of ell and plot the product of all of them as a final result
    :param results: output of moment fitting
    :Nmin: minimal bin of ell in which to fit the Gaussians
    :Nmax: maximal bin of ell in which to fit the Gaussians
    :debug: plot the gaussian fit in each ell to ensure its working well, default: False
    :label: label for the plot
    """
    rl = results['r']
    Nell,N = rl.shape
    K_r = int(2*N**(0.33))
    moytemp=[]
    sigtemp=[]
    for ell in range(Nmin,Nmax):
        y1_cond , bins_cond = np.histogram(rl[ell,:],K_r)
        x1_cond = [.5*(b1+b2) for b1,b2 in zip(bins_cond[:-1],bins_cond[1:])] # Milieu du bin
        ysum_cond = scipy.integrate.simps(y1_cond,x1_cond)
        coeffunit = y1_cond[np.argmax(y1_cond)]/ysum_cond

        pl0=[np.mean(y1_cond),np.std(y1_cond)]
        parinfopl = [{'value':pl0[0], 'fixed':0},{'value':pl0[1],'fixed':0}]
        fa = {'x':x1_cond, 'y':y1_cond/ysum_cond, 'err': 1/(np.sqrt(y1_cond)*ysum_cond)}
        m = mpfit(mpl.Gaussian,parinfo= parinfopl ,functkw=fa,quiet=quiet)
        if m.params[1]>0.01:
            fa = {'x':x1_cond, 'y':y1_cond/ysum_cond, 'err': 1000/(np.sqrt(y1_cond)*ysum_cond)}
            m = mpfit(mpl.Gaussian,parinfo= parinfopl ,functkw=fa,quiet=quiet)        
        if m.params[1]>0.01:
            fa = {'x':x1_cond, 'y':y1_cond/ysum_cond, 'err': 0.0001/(np.sqrt(y1_cond)*ysum_cond)}
            m = mpfit(mpl.Gaussian,parinfo= parinfopl ,functkw=fa,quiet=quiet)            
        
        if debug==True:
            plt.plot(x1_cond,y1_cond/ysum_cond)
            plt.plot(x1_cond,func.Gaussian(x1_cond,m.params[0],m.params[1]))
            plt.show()
        moytemp.append(m.params[0])
        sigtemp.append(m.params[1])
    moy = np.array(moytemp)
    sig = np.array(sigtemp)

    x = np.linspace(-1,1,10000)
    intervall = 0.014
    fig,ax = plt.subplots(1,1, figsize=(10,7))
    gausstot = 1
    for i in range(Nmax-Nmin):
        gausstot = gausstot*func.Gaussian(x,moy[i],sig[i])
    Norm = scipy.integrate.simps(gausstot,x)
    coeffunit = gausstot[np.argmax(gausstot)]/Norm
    pl0=[np.mean(gausstot/Norm/coeffunit),np.std(gausstot/Norm/coeffunit)]
    parinfopl = [{'value':pl0[0], 'fixed':0},{'value':pl0[1],'fixed':0}]
    fa = {'x':x, 'y':gausstot/Norm/coeffunit, 'err': 1000/(np.sqrt(gausstot/Norm))}
    m = mpfit(mpl.Gaussian,parinfo= parinfopl ,functkw=fa,quiet=quiet)        
    if m.params[1]>0.01:
                fa = {'x':x, 'y':gausstot/Norm, 'err': 0.0001/(np.sqrt(gausstot/Norm))}
                m = mpfit(mpl.Gaussian,parinfo= parinfopl ,functkw=fa,quiet=quiet)        
    if m.params[1]>0.01:
                fa = {'x':x, 'y':gausstot/Norm, 'err': 1000/(np.sqrt(gausstot/Norm))}
                m = mpfit(mpl.Gaussian,parinfo= parinfopl ,functkw=fa,quiet=quiet)            
    ax.plot(x,(func.Gaussian(x,m.params[0],m.params[1])/coeffunit)/np.max(func.Gaussian(x,m.params[0],m.params[1])/coeffunit),color=color,linewidth= 5,label='$%s \\pm %s$'%(np.round(m.params[0],5),np.round(m.params[1],5)))
    ax.fill_between(x,(func.Gaussian(x,m.params[0],m.params[1])/coeffunit)/np.max(func.Gaussian(x,m.params[0],m.params[1])/coeffunit),color=color,alpha=0.2,linewidth=5)
    ax.fill_between(x,(func.Gaussian(x,m.params[0],m.params[1])/coeffunit)/np.max(func.Gaussian(x,m.params[0],m.params[1])/coeffunit),facecolor="none",edgecolor=color,linewidth=5)
    #ax.errorbar(x1_cond,y1_cond/ysum_cond/coeffunit,xerr=0,yerr=1/(np.sqrt(y1_cond)*ysum_cond)/coeffunit,fmt='^',color=c[i],ecolor=c[i],zorder=300001,label='$%s \\pm %s$'%(m.params[0],m.params[1]))
    ax.axvline(r, 0, 1, color = 'black', linestyle = "--",linewidth=3,zorder=1)
    ax.plot(x, np.zeros(len(x)), color = 'black', linewidth=5,linestyle='--',zorder=10000000)
    ax.set_xlim([r-intervall,r+intervall])
    ax.legend()
    ax.set_xlabel(r"$\hat{r}$")
    ax.set_ylim([0,1.03])
    if save==True:
        plt.savefig("./plot-gauss/"+kwsave+".pdf")
    else:
        plt.show()
 
# Plot results

def plotmed(ell,label,res,color='darkblue',marker="D",show=True,legend=''):
    ellbound=ell.shape[0]
    name={'A':r'$A^d$','beta':r'$\beta^d$','temp':r'$T^d$','beta_s':r'$\beta^s$','A_s':r'$A^s$','A_sd':r'$A^{sd}$','r':r'$\hat{r}$','X2red':r'$\chi^2$','Aw1b':r'$\mathcal{D}_\ell^{A\times\omega_1^{\beta}}$','Aw1t':r'$\mathcal{D}_\ell^{A\times\omega_1^{T}}$','Asw1bs':r'$\mathcal{D}_\ell^{A_s\times\omega_1^{\beta^s}}$','w1bw1b':r'$\mathcal{D}_\ell^{\omega_1^\beta\times\omega_1^\beta}$','w1tw1t':r'$\mathcal{D}_\ell^{\omega_1^T\times\omega_1^T}$','w1bw1t':r'$\mathcal{D}_\ell^{\omega_1^\beta\times\omega_1^T}$','w1bsw1bs':r'$\mathcal{D}_\ell^{\omega_1^{\beta^s}\times\omega_1^{\beta^s}}$', 'Asw1b':r'$\mathcal{D}_\ell^{A_s\times\omega_1^{\beta}}$','Asw1t':r'$\mathcal{D}_\ell^{A_s\times\omega_1^{T}}$','Adw1s':r'$\mathcal{D}_\ell^{A\times\omega_1^{\beta^s}}$'}
    edgecolor="#80AAF3"
    plt.errorbar(ell,np.median(res[label],axis=1)[:ellbound],yerr=scipy.stats.median_abs_deviation(res[label],axis=1)[:ellbound],c=color,fmt=marker,linestyle='',label=legend)
    plt.scatter(ell,np.median(res[label],axis=1)[:ellbound],s=175,c=color,marker=marker,edgecolor=edgecolor)
    plt.ylabel(name[label],fontsize=20)
    plt.xlabel(r"$\ell$",fontsize=20)
    plt.legend()
    if show==True:
        plt.show()

