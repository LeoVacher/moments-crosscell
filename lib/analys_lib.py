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

def fitmbb(nucross,DL,Linv,p0):
    """
    Fit a mbb and r on a DL
    :param: nucross, array of the cross-frequencies
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param Linv: inverse of the Cholesky matrix
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
            # first mbb fit, dust free, r fixed
            fa = {'x':nucross, 'y':DL[n,:,L], 'err': Linv[L]}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=True)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'r' : paramiterl[:,:,3], 'X2red': chi2l}
    return results


def fito1_b(nucross,DL,Linv,resultsmbb,iter=0):
    """
    Fit using a first order moment expansion in beta on a DL
    :param: nucross, array of the cross-frequencies
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param Linv: inverse of the Cholesky matrix
    :param iter: desired iterations (not impelemented yet)
    :param resultsmbb: must be input mbb best fit in the format of fitmbb()
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
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=True)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'Aw1b' : paramiterl[:,:,3], 'w1bw1b' : paramiterl[:,:,4], 'r' : paramiterl[:,:,5], 'X2red': chi2l}
    return results

def fito1_bT(nucross,DL,Linv,resultsmbb,iter=0):
    """
    Fit using a first order moment expansion in both beta and T on a DL
    :param: nucross, array of the cross-frequencies
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param Linv: inverse of the Cholesky matrix
    :param resultsmbb: must be input mbb best fit in the format of fitmbb()
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
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=True)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'Aw1b' : paramiterl[:,:,3], 'w1bw1b' : paramiterl[:,:,4],'Aw1t' : paramiterl[:,:,5],'w1bw1t' : paramiterl[:,:,6],'w1tw1t' : paramiterl[:,:,7], 'r' : paramiterl[:,:,8], 'X2red': chi2l}
    return results

def fito2_b(nucross,DL,Linv,resultsmbb,iter=0):
    """
    Fit using a second order moment expansion in beta on a DL
    :param: nucross, array of the cross-frequencies
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param Linv: inverse of the Cholesky matrix
    :param resultsmbb: must be input mbb best fit in the format of fitmbb()
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
            # first o1 fit, dust fixed, mom free, r fixed
            parinfopl = [{'value':resultsmbb['A'][L,n], 'fixed':1},{'value':resultsmbb['beta'][L,n], 'fixed':1},{'value':resultsmbb['temp'][L,n], 'fixed':1},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0},{'value':0, 'fixed':0}, {'value':0, 'fixed':0},{'value':L, 'fixed':1}] #dust params
            fa = {'x':nucross, 'y':DL[n,:,L], 'err': Linv[L]}
            m = mpfit(funcfit,parinfo= parinfopl ,functkw=fa,quiet=True)
            paramiterl[L,n]= m.params
            chi2l[L,n]=m.fnorm/m.dof            
    results={'A' : paramiterl[:,:,0], 'beta' : paramiterl[:,:,1], 'temp' : paramiterl[:,:,2], 'Aw1b' : paramiterl[:,:,3], 'w1bw1b' : paramiterl[:,:,4],'Aw2b' : paramiterl[:,:,5],'w1bw2b' : paramiterl[:,:,6],'w2bw2b' : paramiterl[:,:,7], 'r' : paramiterl[:,:,8], 'X2red': chi2l}
    return results

#PLOT FUNCTIONS ##################################################################################################################

def plotr_gaussproduct(results,Nmin=0,Nmax=20,label='MBB',debug=False):
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
    for ell in range(Nell):
        y1_cond , bins_cond = np.histogram(rl[ell,:],K_r)
        x1_cond = [.5*(b1+b2) for b1,b2 in zip(bins_cond[:-1],bins_cond[1:])] # Milieu du bin
        ysum_cond = scipy.integrate.simps(y1_cond,x1_cond)
        coeffunit = y1_cond[np.argmax(y1_cond)]/ysum_cond

        pl0=[np.mean(y1_cond),np.std(y1_cond)]
        parinfopl = [{'value':pl0[0], 'fixed':0},{'value':pl0[1],'fixed':0}]
        fa = {'x':x1_cond, 'y':y1_cond/ysum_cond, 'err': 100000/(np.sqrt(y1_cond)*ysum_cond)}
        m = mpfit(mpl.Gaussian,parinfo= parinfopl ,functkw=fa,quiet=True)
        if debug==True:
            plt.plot(x1_cond,y1_cond/ysum_cond)
            plt.plot(x1_cond,func.Gaussian(x1_cond,m.params[0],m.params[1]))
            plt.show()
        moytemp.append(m.params[0])
        sigtemp.append(m.params[1])
    moy = np.array(moytemp)
    sig = np.array(sigtemp)

    x = np.linspace(-1,1,10000)

    moyfin=[]
    sigfin=[]
    plt.figure(figsize=(30,15))
    gausstot = 1
    for i in range(Nmin,Nmax):
        gausstot = gausstot*func.Gaussian(x,moy[i],sig[i])
    Norm = scipy.integrate.simps(gausstot,x)
    coeffunit = gausstot[np.argmax(gausstot)]/Norm
    pl0=[np.mean(gausstot/Norm/coeffunit),np.std(gausstot/Norm/coeffunit)]
    parinfopl = [{'value':pl0[0], 'fixed':0},{'value':pl0[1],'fixed':0}]
    fa = {'x':x, 'y':gausstot/Norm/coeffunit, 'err': 1000/(np.sqrt(gausstot/Norm))}
    m = mpfit(mpl.Gaussian,parinfo= parinfopl ,functkw=fa)        
    if m.params[1]>0.01:
        fa = {'x':x, 'y':gausstot/Norm, 'err': 0.0001/(np.sqrt(gausstot/Norm))}
        m = mpfit(mpl.Gaussian,parinfo= parinfopl ,functkw=fa)        

    plt.xlim([-0.015,0.015])
    plt.plot(x,gausstot/Norm/coeffunit,label=r"%s $\mu =%s$ $\sigma = %s$"%(label,np.round(m.params[0],6),np.round(m.params[1],6)),linewidth=1.5,color="blue")
    plt.axvline( m.params[0],0, 1, color = 'blue', linestyle = "-.",alpha=0.7)
    plt.axvline( m.params[0]+m.params[1],0, 1, color = 'blue', linestyle = "-.",alpha=0.7)
    plt.axvline( m.params[0]-m.params[1],0, 1, color = 'blue', linestyle = "-.",alpha=0.7)
    plt.axvline(0, 0, 1, color = 'red', linestyle = "--")
    plt.xlabel("r",fontsize=40)
    plt.legend()
    plt.show()

