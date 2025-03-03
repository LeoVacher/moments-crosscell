import numpy as np
import scipy
from collections import Counter


#contains all function for covariance computations, develloped in collaboration with S. Vinzl.

#matrix manipulations:

def max_rep(arr):
    """
    count the maximal number of repetitions in an array or list
    :param arr: numpy array or list
    """
    counts = Counter(arr)  
    return max(counts.values()) 

def cross_index(A, B, Nf):
    """
    find the index of the cross-spectra associated to the band doublet (A,B)
    :param A,B: indices of bands A and B
    :param Nf: number of frequencies
    """
    if A > B:
        A, B = B, A 
    return int((A * (2 * Nf - A + 1)) // 2 + (B - A))

def covtocorr(cov):
    """
    compute correlation matrix associated to a covariance
    :param cov: covariance matrix
    """
    std_dev = np.sqrt(np.diag(cov)) 
    outer_std = np.outer(std_dev, std_dev) 
    corr = cov / outer_std
    return corr

def block_diag(diaglist):
    '''
    Build a covariance matrix with diaglist in each block diagonal
    '''
    return np.block([[diaglist[i] if i == j else np.zeros_like(diaglist[0]) 
                      for j in range(len(a_list))] 
                      for i in range(len(a_list))])

def is_cholesky_possible(cov):
    """Is Cholesky working?"""
    try:
        inv=np.linalg.inv(cov)
        np.linalg.cholesky(inv)
        return True  
    except np.linalg.LinAlgError:
        return False 

def nearest_PSD(cov):
    """
    Compute the nearest positive semi-definite matrix of a symmetric matrix
    """
    eigenvals, eigenvects = np.linalg.eig(cov)
    eigenvals[eigenvals < 0] = 0

    Q = eigenvects
    D_plus = np.diag(eigenvals)

    nPSD = Q @ D_plus @ Q.T
    nPSD = (nPSD + nPSD.T) / 2

    return nPSD

#compute covariances:

def getLinvdiag(DL,printdiag=False,offset=0):
    """
    Compute inverse of the covariance matrix used for the fit assuming it is block-diagonal in ell. 
    :param DL: The input binned DL array should be of the shape (Nsim, Ncross, Nell)
    :param print: if true, print the diagonal of cov.invcov to evaluate the quality of the inversion.
    :return Linv: Cholesky matrix in the shape (Nell,ncross,ncross)
    """
    _, _, Nell = DL.shape
    DLtempo = np.swapaxes(DL, 0, 1)
    Linvdc = []
    for L in range(Nell):
        cov = np.cov(DLtempo[:,:,L])
        invcov = np.linalg.inv(cov + offset * np.identity(len(cov)))        
        if printdiag:
            print(np.diag(np.dot(cov, invcov)))            
        Linvdc.append(np.linalg.cholesky(invcov))
    return np.array(Linvdc)

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

def cov_analytic(A,B,C,D,DLcross_fg=None,DL_cross_lens=None,DL_cross_noise=None,ell=None,Nlbin=None,mask=None,corrfog=False):
    """
    compute the analytical estimate of the covariance from the signal using Knox formula
    :param nu_i,nu_j,nu_k,nu_l: The quadruplet of frequencies (nu_i,nu_j),(nu_k,nu_l) for which the covariance should be computed in GHz.
    :param DLcross_fg: The foreground binned DL array should be of the shape (Ncross, Nell). Needed only to compute the analytical Knox formula (type=Knox-fg and type=Knox+fg)
    :param DL_cross_lens: The cmb binned DL array should be of the shape (Ncross, Nell). Needed only to compute the analytical Knox formula (type=Knox-fg and type=Knox+fg)
    :param DL_cross_noise: The noise binned DL array should be of the shape (Ncross, Nell). Needed only to compute the analytical Knox formula (type=Knox-fg and type=Knox+fg)
    :param corrfog: if true correct for the cosmic variance of the foregrounds.
    """
    Nf= int((np.sqrt(1 + 8 * DLcross_fg.shape[0])-1)/2)

    bands = np.array([A, B, C, D])
    counter = Counter(bands)

    poscrossAA= cross_index(A, A, Nf)
    poscrossBB= cross_index(B, B, Nf)
    poscrossAB= cross_index(A, B, Nf)
    poscrossCD= cross_index(C, D, Nf)
    poscrossAC= cross_index(A, C, Nf)
    poscrossBD= cross_index(B, D, Nf)
    poscrossAD= cross_index(A, D, Nf)
    poscrossBC= cross_index(B, C, Nf)


    v_l = (2*ell+1)*Nlbin*np.mean(mask**2)**2/np.mean(mask**4)
    
    if A ==C and B == D:
        if A==B:
            DLAB =  DL_cross_lens[poscrossAA] + DLcross_fg[poscrossAA]
            DLAA =  DL_cross_lens[poscrossAA] + 2*DL_cross_noise[poscrossAA] + DLcross_fg[poscrossAA] 
            DLBB =  DLAA
        else:
            DLAB = DLcross_fg[poscrossAB] + DL_cross_lens[poscrossAB]
            DLAA = DL_cross_lens[poscrossAA] + DL_cross_noise[poscrossAA] +DLcross_fg[poscrossAA]
            DLBB = DL_cross_lens[poscrossBB] + DL_cross_noise[poscrossBB] +DLcross_fg[poscrossBB]
        
        if corrfog==True:
            covmat = (DLAB**2+DLAA*DLBB - DLcross_fg[poscrossAB]**2 -DLcross_fg[poscrossAA]*DLcross_fg[poscrossBB])/v_l
        else:
            covmat = (DLAB**2+DLAA*DLBB)/v_l

    elif max(counter.values()) == 2 and A != B and C != D:
        rep_band = np.array(list(counter))[np.where(np.array(list(counter.values())) == 2)][0]
        rep_ind = np.where(bands == rep_band)[0]

        if all(rep_ind == [1, 2]):
            bands[0], bands[1] = bands[1], bands[0]

        elif all(rep_ind == [0, 3]):
            bands[2], bands[3] = bands[3], bands[2]

        elif all(rep_ind == [1, 3]):
            bands[0], bands[1] = bands[1], bands[0]
            bands[2], bands[3] = bands[3], bands[2]

        A, B, C, D = bands

        poscrossAC = cross_index(A, C, Nf)
        poscrossBD = cross_index(B, D, Nf)
        poscrossAD = cross_index(A, D, Nf)
        poscrossBC = cross_index(B, C, Nf)

        DLAC = DLcross_fg[poscrossAC] + DL_cross_lens[poscrossAC] + DL_cross_noise[poscrossAC]
        DLBD = DLcross_fg[poscrossBD] + DL_cross_lens[poscrossBD]
        DLAD = DLcross_fg[poscrossAD] + DL_cross_lens[poscrossAD]
        DLBC = DLcross_fg[poscrossBC] + DL_cross_lens[poscrossBC]

        if corrfog == True:
            covmat = (DLAC * DLBD + DLAD * DLBC - DLcross_fg[poscrossAC] * DLcross_fg[poscrossBD] - DLcross_fg[poscrossAD] * DLcross_fg[poscrossBC]) / v_l
        else:
            covmat = (DLAC * DLBD + DLAD * DLBC) / v_l

    else:
        DLAC= DLcross_fg[poscrossAC] + DL_cross_lens[poscrossAC]
        DLBD= DLcross_fg[poscrossBD] + DL_cross_lens[poscrossBD]
        DLAD= DLcross_fg[poscrossAD] + DL_cross_lens[poscrossAD]
        DLBC= DLcross_fg[poscrossBC] + DL_cross_lens[poscrossBC]
        if corrfog==True:
            covmat = (DLAC*DLBD+DLAD*DLBC - DLcross_fg[poscrossAC]*DLcross_fg[poscrossBD] - DLcross_fg[poscrossAD]*DLcross_fg[poscrossBC])/v_l
        else: 
            covmat = (DLAC*DLBD+DLAD*DLBC)/v_l
    return covmat

def cov_analytic_signal(A,B,C,D,DL_signal=None,ell=None,Nlbin=None,mask=None):
    """
    compute the analytical estimate of the covariance from the signal using Knox formula
    :param A,B,C,D: The quadruplet of frequency bands (A,B),(C,D) for which the covariance should be computed in GHz.
    :param DL_signal: The signal binned DL array should be of the shape (Nsim, Ncross, Nell).
    """
    Nf= int((np.sqrt(1 + 8 * DL_signal.shape[1])-1)/2) 
    poscrossAC= cross_index(A, C, Nf)
    poscrossBD= cross_index(B, D, Nf)
    poscrossAD= cross_index(A, D, Nf)
    poscrossBC= cross_index(B, C, Nf)

    v_l = (2*ell+1)*Nlbin*np.mean(mask**2)**2/np.mean(mask**4)

    DLAC= DL_signal[0,poscrossAC] 
    DLBD= DL_signal[0,poscrossBD]
    DLAD= DL_signal[0,poscrossAD]
    DLBC= DL_signal[0,poscrossBC]
    
    covmat= (DLAC*DLBD+DLAD*DLBC)/v_l
    return covmat 

def cov_NaMaster(A, B, C, D, Cls_cmb_EE, Cls_cmb_BB, Cls_fg_EE, Cls_fg_BB, Nls_EE, Nls_BB, mask, w, corrfog=True, output='all'):
    """
    Compute analytical covariance matrix using NaMaster fonctions from theoretical power spectra and noise model.
    Cls can be mode-decoupled unbinned power spectra, but better accuracy is achieved if they are mode-coupled pseudo-Cls divided by fsky.
    :param A,B,C,D: The quadruplet of frequency bands (A,B),(C,D) for which the covariance should be computed.
    :param Cls_cmb_EE: CMB EE power spectra. Must be of dimension (Ncross, 3*NSIDE).
    :param Cls_cmb_BB: CMB BB power spectra. Must be of dimension (Ncross, 3*NSIDE).
    :param Cls_fg_EE: Foregrounds EE power spectra. Must be of dimension (Ncross, 3*NSIDE).
    :param Cls_fg_BB: Foregrounds BB power spectra. Must be of dimension (Ncross, 3*NSIDE).
    :param Nls_EE: Noise EE power spectra. Must be of dimension (Ncross, 3*NSIDE).
    :param Nls_BB: Noise BB power spectra. Must be of dimension (Ncross, 3*NSIDE).
    :param mask: Mask used for computing the power spectra.
    :param w: NmtWorkspace used for computing the power spectra.
    :param corrfog: If True, correct for the cosmic variance of foregrounds. Default: True.
    :param output: If 'EE', compute covariance matrix for E modes only. If 'BB', compute covariance matrix for B modes only. If 'all', compute the full covariance matrix. Default: 'all'.
    """
    Nfreqs = int((np.sqrt(1 + 8*Cls_cmb_EE.shape[0]) - 1) / 2)
    
    f = nmt.NmtField(mask, None, spin=2)
    cw = nmt.NmtCovarianceWorkspace()
    cw.compute_coupling_coefficients(f, f)
    lmax = cw.wsp.lmax
            
    crossAC = cross_index(A, C, Nfreqs)
    crossBD = cross_index(B, D, Nfreqs)
    crossAD = cross_index(A, D, Nfreqs)
    crossBC = cross_index(B, C, Nfreqs)
            
    bands = np.array([bandA, bandB, bandC, bandD])
    counter = Counter(bands)
            
    if A == C and B == D:
        if A == B:
            EE_a1b1 = Cls_cmb_EE[crossAC] + Cls_fg_EE[crossAC] + 2*Nls_EE[crossAC]
            EE_a1b2 = Cls_cmb_EE[crossAD] + Cls_fg_EE[crossAD]
            EE_a2b1 = Cls_cmb_EE[crossBC] + Cls_fg_EE[crossBC]
            EE_a2b2 = Cls_cmb_EE[crossBD] + Cls_fg_EE[crossBD] + 2*Nls_EE[crossBD]
                    
            BB_a1b1 = Cls_cmb_BB[crossAC] + Cls_fg_BB[crossAC] + 2*Nls_BB[crossAC]
            BB_a1b2 = Cls_cmb_BB[crossAD] + Cls_fg_BB[crossAD]
            BB_a2b1 = Cls_cmb_BB[crossBC] + Cls_fg_BB[crossBC]
            BB_a2b2 = Cls_cmb_BB[crossBD] + Cls_fg_BB[crossBD] + 2*Nls_BB[crossBD]
                    
        else:
            EE_a1b1 = Cls_cmb_EE[crossAC] + Cls_fg_EE[crossAC] + Nls_EE[crossAC]
            EE_a1b2 = Cls_cmb_EE[crossAD] + Cls_fg_EE[crossAD]
            EE_a2b1 = Cls_cmb_EE[crossBC] + Cls_fg_EE[crossBC]
            EE_a2b2 = Cls_cmb_EE[crossBD] + Cls_fg_EE[crossBD] + Nls_EE[crossBD]
                    
            BB_a1b1 = Cls_cmb_BB[crossAC] + Cls_fg_BB[crossAC] + Nls_BB[crossAC]
            BB_a1b2 = Cls_cmb_BB[crossAD] + Cls_fg_BB[crossAD]
            BB_a2b1 = Cls_cmb_BB[crossBC] + Cls_fg_BB[crossBC]
            BB_a2b2 = Cls_cmb_BB[crossBD] + Cls_fg_BB[crossBD] + Nls_BB[crossBD]
                    
    elif max(counter.values()) == 2 and A != B and C != D:
        rep_band = np.array(list(counter))[np.where(np.array(list(counter.values())) == 2)][0]
        rep_ind = np.where(bands == rep_band)[0]
                    
        if all(rep_ind == [1,2]):
            bands[0], bands[1] = bands[1], bands[0]
                    
        elif all(rep_ind == [0,3]):
            bands[2], bands[3] = bands[3], bands[2]
                    
        elif all(rep_ind == [1,3]):
            bands[0], bands[1] = bands[1], bands[0]
            bands[2], bands[3] = bands[3], bands[2]
                    
        A, B, C, D = bands
                
        crossAC = cross_index(A, C, Nfreqs)
        crossBD = cross_index(B, D, Nfreqs)
        crossAD = cross_index(A, D, Nfreqs)
        crossBC = cross_index(B, C, Nfreqs)
                
        EE_a1b1 = Cls_cmb_EE[crossAC] + Cls_fg_EE[crossAC] + Nls_EE[crossAC]
        EE_a1b2 = Cls_cmb_EE[crossAD] + Cls_fg_EE[crossAD]
        EE_a2b1 = Cls_cmb_EE[crossBC] + Cls_fg_EE[crossBC]
        EE_a2b2 = Cls_cmb_EE[crossBD] + Cls_fg_EE[crossBD]
                
        BB_a1b1 = Cls_cmb_BB[crossAC] + Cls_fg_BB[crossAC] + Nls_BB[crossAC]
        BB_a1b2 = Cls_cmb_BB[crossAD] + Cls_fg_BB[crossAD]
        BB_a2b1 = Cls_cmb_BB[crossBC] + Cls_fg_BB[crossBC]
        BB_a2b2 = Cls_cmb_BB[crossBD] + Cls_fg_BB[crossBD]
                
    else:
        EE_a1b1 = Cls_cmb_EE[crossAC] + Cls_fg_EE[crossAC]
        EE_a1b2 = Cls_cmb_EE[crossAD] + Cls_fg_EE[crossAD]
        EE_a2b1 = Cls_cmb_EE[crossBC] + Cls_fg_EE[crossBC]
        EE_a2b2 = Cls_cmb_EE[crossBD] + Cls_fg_EE[crossBD]
                
        BB_a1b1 = Cls_cmb_BB[crossAC] + Cls_fg_BB[crossAC]
        BB_a1b2 = Cls_cmb_BB[crossAD] + Cls_fg_BB[crossAD]
        BB_a2b1 = Cls_cmb_BB[crossBC] + Cls_fg_BB[crossBC]
        BB_a2b2 = Cls_cmb_BB[crossBD] + Cls_fg_BB[crossBD]
                
    Cl_a1b1 = [EE_a1b1, np.zeros(lmax+1), np.zeros(lmax+1), BB_a1b1]
    Cl_a1b2 = [EE_a1b2, np.zeros(lmax+1), np.zeros(lmax+1), BB_a1b2]
    Cl_a2b1 = [EE_a2b1, np.zeros(lmax+1), np.zeros(lmax+1), BB_a2b1]
    Cl_a2b2 = [EE_a2b2, np.zeros(lmax+1), np.zeros(lmax+1), BB_a2b2]
            
    covmat = nmt.gaussian_covariance(cw, 2, 2, 2, 2, Cl_a1b1, Cl_a1b2, Cl_a2b1, Cl_a2b2, w)
            
    if corrfog:
        EE_a1b1 = Cls_cmb_EE[crossAC]
        EE_a1b2 = Cls_cmb_EE[crossAD]
        EE_a2b1 = Cls_cmb_EE[crossBC]
        EE_a2b2 = Cls_cmb_EE[crossBD]
                
        BB_a1b1 = Cls_cmb_BB[crossAC]
        BB_a1b2 = Cls_cmb_BB[crossAD]
        BB_a2b1 = Cls_cmb_BB[crossBC]
        BB_a2b2 = Cls_cmb_BB[crossBD]
                
        Cl_a1b1 = [EE_a1b1, np.zeros(lmax+1), np.zeros(lmax+1), BB_a1b1]
        Cl_a1b2 = [EE_a1b2, np.zeros(lmax+1), np.zeros(lmax+1), BB_a1b2]
        Cl_a2b1 = [EE_a2b1, np.zeros(lmax+1), np.zeros(lmax+1), BB_a2b1]
        Cl_a2b2 = [EE_a2b2, np.zeros(lmax+1), np.zeros(lmax+1), BB_a2b2]
                
        covmat -= nmt.gaussian_covariance(cw, 2, 2, 2, 2, Cl_a1b1, Cl_a1b2, Cl_a2b1, Cl_a2b2, w)
            
    covmat = cov.reshape([Nbins, 4, Nbins, 4])
            
    if output == 'EE':
        return covmat[:, 0, :, 0]
            
    elif output == 'BB':
        return covmat[:, 3, :, 3]
            
    else:
        return covmat
    
def cov_NaMaster_signal(A, B, C, D, Cls_EE, Cls_BB, mask, w, output='all'):
    """
    Compute analytical covariance matrix using NaMaster fonctions directly from signal.
    Cls can be mode-decoupled unbinned power spectra, but better accuracy is achieved if they are mode-coupled pseudo-Cls divided by fsky.
    :param A,B,C,D: The quadruplet of frequency bands (A,B),(C,D) for which the covariance should be computed.
    :param Cls_EE: EE cross-power spectra. Must be of dimension (Ncross, 3*NSIDE).
    :param Cls_BB: BB cross-power spectra. Must be of dimension (Ncross, 3*NSIDE).
    :param mask: Mask used for computing the power spectra.
    :param w: NmtWorkspace used for computing the power spectra.
    :param output: If 'EE', compute covariance matrix for E modes only. If 'BB', compute covariance matrix for B modes only. If 'all', compute the full covariance matrix. Default: 'all'.
    """
    Nfreqs = int((np.sqrt(1 + 8*Cls_EE.shape[0]) - 1) / 2)
    
    f = nmt.NmtField(mask, None, spin=2)
    cw = nmt.NmtCovarianceWorkspace()
    cw.compute_coupling_coefficients(f, f)
    lmax = cw.wsp.lmax
            
    crossAC = cross_index(A, C, Nfreqs)
    crossBD = cross_index(B, D, Nfreqs)
    crossAD = cross_index(A, D, Nfreqs)
    crossBC = cross_index(B, C, Nfreqs)
            
    EE_a1b1 = Cls_EE[crossAC]
    EE_a1b2 = Cls_EE[crossAD]
    EE_a2b1 = Cls_EE[crossBC]
    EE_a2b2 = Cls_EE[crossBD]
            
    BB_a1b1 = Cls_BB[crossAC]
    BB_a1b2 = Cls_BB[crossAD]
    BB_a2b1 = Cls_BB[crossBC]
    BB_a2b2 = Cls_BB[crossBD]
            
    Cl_a1b1 = [EE_a1b1, np.zeros(lmax+1), np.zeros(lmax+1), BB_a1b1]
    Cl_a1b2 = [EE_a1b2, np.zeros(lmax+1), np.zeros(lmax+1), BB_a1b2]
    Cl_a2b1 = [EE_a2b1, np.zeros(lmax+1), np.zeros(lmax+1), BB_a2b1]
    Cl_a2b2 = [EE_a2b2, np.zeros(lmax+1), np.zeros(lmax+1), BB_a2b2]
            
    covmat = nmt.gaussian_covariance(cw, 2, 2, 2, 2, Cl_a1b1, Cl_a1b2, Cl_a2b1, Cl_a2b2, w)
    covmat = cov.reshape([Nbins, 4, Nbins, 4])
            
    if output == 'EE':
        return = covmat[:, 0, :, 0]
            
    elif output == 'BB':
        return covmat[:, 3, :, 3]
            
    else:
        return covmat

def compute_analytical_cov(DL_signal=None,DLcross_fg=None,DL_cross_lens=None,DL_cross_noise=None,type='signal',ell=None,Nlbin=None,mask=None,Linv=True):
    """
    compute an analytical estimate of the covariance matrix in different fashion (see Tristram+2004 arxiv:0405575 Eq.28).
    :param DL_signal: The signal binned DL array should be of the shape (Nsim, Ncross, Nell). Needed only to compute Knox formula from the signal (type=signal)
    :param DLcross_fg: The foreground binned DL array should be of the shape (Ncross, Nell). Needed only to compute the analytical Knox formula (type=Knox-fg and type=Knox+fg)
    :param DL_cross_lens: The cmb binned DL array should be of the shape (Ncross, Nell). Needed only to compute the analytical Knox formula (type=Knox-fg and type=Knox+fg)
    :param DL_cross_noise: The noise binned DL array should be of the shape (Ncross, Nell). Needed only to compute the analytical Knox formula (type=Knox-fg and type=Knox+fg)
    :param type: type of estimate of the covariance. Can be "signal", "Knox-fg" or "Knox+fg"
    :param Linv: If true return the Cholesky matrix computed from the inverse cov. If false return the covariance matrix. 
    """
    N, Ncross, Nell= DL_signal.shape
    N_freqs= int((np.sqrt(1 + 8*Ncross)-1)/2)
    covmat = np.zeros((Nell,Ncross,Ncross))
    doublets = {}
    
    z=0
    for i in range(0,N_freqs):
        for j in range(i,N_freqs):
            doublets[z]=(i, j)
            z=z+1
        
    for i in range(0,Ncross):
        for j in range(0,Ncross):
            A,B= doublets[i]
            C,D= doublets[j]
            if type=='signal':
                covmat[:,i,j]= cov_analytic_signal(A,B,C,D,DL_signal,ell=ell,Nlbin=Nlbin,mask=mask)
            if type=='Knox+fg':
                covmat[:,i,j]= cov_analytic(A,B,C,D,DL_cross_lens=DL_cross_lens,DLcross_fg=DLcross_fg,DL_cross_noise=DL_cross_noise,ell=ell,Nlbin=Nlbin,mask=mask,corrfog=False)
            if type=='Knox-fg':
                covmat[:,i,j]= cov_analytic(A,B,C,D,DL_cross_lens=DL_cross_lens,DLcross_fg=DLcross_fg,DL_cross_noise=DL_cross_noise,ell=ell,Nlbin=Nlbin,mask=mask,corrfog=True)
    if Linv==True:
        Linv = np.zeros((Nell,Ncross,Ncross))
        invcov = np.zeros((Nell,Ncross,Ncross))
        for L in range(Nell):
            cov = np.copy(covmat[L])
            mean_diag = np.mean(np.diag(cov))
            offset_L=np.log10(mean_diag)-15
            while is_cholesky_possible(cov)==False:
                cov= cov+ 10**(offset_L)*np.identity(len(cov))
                offset_L=offset_L+1

                if offset_L >= np.log10(0.01 * mean_diag):
                    cov = nearest_PSD(covmat[L])
                    mean_diag = np.mean(np.diag(cov))
                    offset_L = np.log10(mean_diag) - 15

                    while is_cholesky_possible(cov) == False:
                        cov += 10**(offset_L) * np.identity(len(cov))
                        offset_L += 1

                        if offset_L >= np.log10(0.01 * mean_diag):
                            raise ValueError(f"Error: the diagonal is altered by more than 1% in the bin L={L}")

            covmat[L] = cov
            invcov[L]=np.linalg.inv(covmat[L])
            Linv[L]= np.linalg.cholesky(invcov[L])
        return Linv, invcov
    else:
        return covmat
