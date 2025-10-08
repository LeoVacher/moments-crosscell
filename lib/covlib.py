import numpy as np
import healpy as hp
import pymaster as nmt
from collections import Counter
from tqdm import tqdm


# Contains all functions for covariance computations


"""
Matrix manipulations
"""

def cross_index(A, B, Nf):
    """
    Find the index of the cross-spectrum associated to the band doublet (A, B).

    Parameters
    ----------
    A : int
        First band associated to cross-spectrum. Must be in interval [0, Nf[
    B : int
        Second band associated to cross-spectrum. Must be in interval [0, Nf[
    Nf : int
        Number of frequency bands.

    Returns
    -------
    int
        Index of the associated cross-spectrum.

    """
    if A > B:
        A, B = B, A 
    return int((A * (2 * Nf - A + 1)) // 2 + (B - A))


def band_doublet(Nf):
    """
    Find the band doublet associated to all the input cross-spectrum indices.

    Parameters
    ----------
    Nf : int
        Number of frequency bands.

    Returns
    -------
    np.array
        Band doublet associated to all cross-spectra indices.
    """
    doublets = [(i, j) for i in range(Nf) for j in range(i, Nf)]
    return doublets

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
                      for j in range(len(diaglist))] 
                      for i in range(len(diaglist))])

def is_positive_definite(M):
    """
    Check if a matrix M is positive definite.

    Parameters
    ----------
    M : np.array
        Matrix to check.

    Returns
    -------
    bool
        True if M is positive definite, otherwise False.

    """
    try:
        np.linalg.cholesky(M)
        return True
    
    except np.linalg.LinAlgError:
        return False
    
def nearest_PSD(M):
    """
    Compute the nearest positive semi-definite matrix of a symmetric matrix M

    Parameters
    ----------
    M : np.array
        Matrix to compute the nearest positive semi-definite.

    Returns
    -------
    np.array
        Computed nearest positive semi-definite matrix.

    """
    eigenvals, eigenvects = np.linalg.eig(M)
    eigenvals[eigenvals < 0] = 0
    
    Q = eigenvects
    D_plus = np.diag(eigenvals)
    
    nSPD = Q @ D_plus @ Q.T
    nSPD = (nSPD + nSPD.T) / 2
    
    return nSPD

def compute_inverse(M):
    """
    Compute inverse of a matrix M to the best achievable accuracy.

    Parameters
    ----------
    M : np.array
        Matrix to inverse.

    Raises
    ------
    ValueError
        Raises error if M is not invertible.

    Returns
    -------
    inv : np.array
        Inversed matrix.
    new : np.array
        Altered matrix such as the computed inverse is positive definite.

    """
    mean_diag = np.mean(np.diag(M))
    offset = 10**(np.log10(mean_diag) - 15)
    
    new = M
    
    while is_positive_definite(np.linalg.inv(new)) == False:
        new += offset * np.identity(len(M))
        offset *= 10
        
        if offset >= 0.01 * mean_diag:
            new = nearest_PSD(M)
            mean_diag = np.mean(np.diag(new))
        
            offset = 10**(np.log10(mean_diag) - 15)
            
            while is_positive_definite(np.linalg.inv(new)) == False:
                new += offset * np.identity(len(M))
                offset *= 10
                
                if offset >= 0.01 * mean_diag:
                    raise ValueError('Diagonal is altered of more than 1% even with nearest positive semi-definite')
                    
    inv = np.linalg.inv(new)
    
    return inv, new

"""
Covariance computations
"""

def getLinvdiag(DL,printdiag=False,offset=0):
    # old function to be improved
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
    # old function to be improved
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


def cov_Knox(mask, Cls_cmb, Cls_fg, Nls, w, corfg=True, progress=False):
    """
    Compute analytical covariance matrix using Knox formula from theoretical power spectra and noise model.
    Cls can be mode-decoupled power spectra, but better accuracy is achieved if they are mode-coupled pseudo-Cls divided by fsky.

    Parameters
    ----------
    mask : np.array
        Mask used for computing the power spectra.
    Cls_cmb : np.array
        CMB power spectra. Must be of dimension (Ncross, Nbins).
    Cls_fg : np.array
        Foregrounds power spectra. Must be of dimension (Ncross, Nbins).
    Nls : np.array
        Noise power spectra. Must be of dimension (Ncross, Nbins).
    w : nmt.NmtWorkspace
        Workspace used for computing the power spectra.
    corfg : bool, optional
        If True, correct for the cosmic variance of foregrounds. Default: True.
    progress : bool, optional
        If True, display a progress bar while computing the covariance matrix. Default: False.

    Returns
    -------
    np.array
        Computed covariance matrix.

    """
    lmax = w.wsp.lmax
    Ncross, Nbins = Cls_cmb.shape
    delta_l = int(lmax / Nbins)
    Nfreqs = int((np.sqrt(1 + 8*Ncross) - 1) / 2)
    doublets = band_doublet(Nfreqs)
    
    b = nmt.NmtBin.from_lmax_linear(lmax, nlb=delta_l)
    leff = b.get_effective_ells()
    
    nu_l = (2*leff+1) * delta_l * np.mean(mask**2)**2 / np.mean(mask**4)
    
    covmat = np.zeros((Nbins*Ncross, Nbins*Ncross))
    
    if progress:
        pbar = tqdm(desc='Estimating covariance matrix', total=int(Ncross*(Ncross+1)/2))
    
    for crossAB in range(Ncross):
        for crossCD in range(crossAB, Ncross):
            A, B = doublets[crossAB]
            C, D = doublets[crossCD]
            
            crossAC = cross_index(A, C, Nfreqs)
            crossBD = cross_index(B, D, Nfreqs)
            crossAD = cross_index(A, D, Nfreqs)
            crossBC = cross_index(B, C, Nfreqs)
            
            bands = np.array([A, B, C, D])
            counter = Counter(bands)
            
            if A == C and B == D:
                if A == B:
                    Cl_AC = Cls_cmb[crossAC] + Cls_fg[crossAC] + 2*Nls[crossAC]
                    Cl_BD = Cls_cmb[crossBD] + Cls_fg[crossBD] + 2*Nls[crossBD]
                    Cl_AD = Cls_cmb[crossAD] + Cls_fg[crossAD]
                    Cl_BC = Cls_cmb[crossBC] + Cls_fg[crossBC]
                
                else:
                    Cl_AC = Cls_cmb[crossAC] + Cls_fg[crossAC] + Nls[crossAC]
                    Cl_BD = Cls_cmb[crossBD] + Cls_fg[crossBD] + Nls[crossBD]
                    Cl_AD = Cls_cmb[crossAD] + Cls_fg[crossAD]
                    Cl_BC = Cls_cmb[crossBC] + Cls_fg[crossBC]
                    
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
                
                Cl_AC = Cls_cmb[crossAC] + Cls_fg[crossAC] + Nls[crossAC]
                Cl_BD = Cls_cmb[crossBD] + Cls_fg[crossBD]
                Cl_AD = Cls_cmb[crossAD] + Cls_fg[crossAD]
                Cl_BC = Cls_cmb[crossBC] + Cls_fg[crossBC]
                
            else:
                Cl_AC = Cls_cmb[crossAC] + Cls_fg[crossAC]
                Cl_BD = Cls_cmb[crossBD] + Cls_fg[crossBD]
                Cl_AD = Cls_cmb[crossAD] + Cls_fg[crossAD]
                Cl_BC = Cls_cmb[crossBC] + Cls_fg[crossBC]

            cov = 1/nu_l * (Cl_AC * Cl_BD + Cl_AD * Cl_BC)
            
            if corfg:
                cov -= 1/nu_l * (Cls_fg[crossAC] * Cls_fg[crossBD] + Cls_fg[crossAD] * Cls_fg[crossBC])

            for i in range(Nbins):
                covmat[i*Ncross + crossAB, i*Ncross + crossCD] = cov[i]
                
                if crossAB != crossCD:
                    covmat[i*Ncross + crossCD, i*Ncross + crossAB] = cov[i]
                    
            if progress:
                pbar.update(1)
                
    if progress:
        pbar.close()
    
    return covmat

def cov_Knox_signal(mask, Cls, w, progress=False):
    """
    Compute analytical covariance matrix using Knox formula directly from signal.
    Cls can be mode-decoupled power spectra, but better accuracy is achieved if they are binned mode-coupled pseudo-Cls divided by fsky.

    Parameters
    ----------
    mask : np.array
        Mask used for computing the power spectra.
    Cls : np.array
        Cross-power spectra. Must be of dimension (Ncross, Nbins).
    w : nmt.NmtWorkspace
        Workspace used for computing the power spectra.
    progress : bool, optional
        If True, display a progress bar while computing the covariance matrix. Default: False.

    Returns
    -------
    np.array
        Computed covariance matrix.

    """
    lmax = w.wsp.lmax
    Ncross, Nbins = Cls.shape
    delta_l = int(lmax / Nbins)
    Nfreqs = int((np.sqrt(1 + 8*Ncross) - 1) / 2)
    doublets = band_doublet(Nfreqs)
    
    b = nmt.NmtBin.from_lmax_linear(lmax, nlb=delta_l)
    leff = b.get_effective_ells()
    
    nu_l = (2*leff+1) * delta_l * np.mean(mask**2)**2 / np.mean(mask**4)
    
    covmat = np.zeros((Nbins*Ncross, Nbins*Ncross))
    
    if progress:
        pbar = tqdm(desc='Estimating covariance matrix', total=int(Ncross*(Ncross+1)/2))
    
    for crossAB in range(Ncross):
        for crossCD in range(crossAB, Ncross):
            A, B = doublets[crossAB]
            C, D = doublets[crossCD]
            
            crossAC = cross_index(A, C, Nfreqs)
            crossBD = cross_index(B, D, Nfreqs)
            crossAD = cross_index(A, D, Nfreqs)
            crossBC = cross_index(B, C, Nfreqs)
            
            cov = 1/nu_l * (Cls[crossAC] * Cls[crossBD] + Cls[crossAD] * Cls[crossBC])
            
            for i in range(Nbins):
                covmat[i*Ncross + crossAB, i*Ncross + crossCD] = cov[i]
                
                if crossAB != crossCD:
                    covmat[i*Ncross + crossCD, i*Ncross + crossAB] = cov[i]
                
            if progress:
                pbar.update(1)
                
    if progress:
        pbar.close()
        
    return covmat

def cov_NaMaster(mask, Cls_cmb, Cls_fg, Nls, w, corfg=True, output='all', progress=False):
    """
    Compute analytical covariance matrix using NaMaster fonctions from theoretical power spectra and noise model.
    Cls can be mode-decoupled unbinned power spectra, but better accuracy is achieved if they are mode-coupled pseudo-Cls divided by fsky.

    Parameters
    ----------
    mask : np.array
        Mask used for computing the power spectra.
    Cls_cmb : list
        CMB power spectra. Must contain 3 numpy arrays of dimension (Ncross, 3*NSIDE) corresponding to TT, EE and BB spectra.
    Cls_fg : list
        Foregrounds power spectra. Must contain 3 numpy arrays of dimension (Ncross, 3*NSIDE) corresponding to TT, EE and BB spectra.
    Nls : list
        Noise power spectra. Must contain 3 numpy arrays of dimension (Ncross, 3*NSIDE) corresponding to TT, EE and BB spectra.
    w : list
        List of nmt.NmtWorkspace used for computing TT, EE and BB power spectra.
    corfg : bool, optional
        If True, correct for the cosmic variance of foregrounds. Default: True.
    output : string, optional
        If 'TT', return covariance matrix for temperature.
        If 'EE', return covariance matrix for E-modes.
        If 'BB', return covariance matrix for B-modes.
        If 'all', return covariance matrices of TT, EE, BB spectra.
        Default: 'all'.
    progress : bool, optional
        If True, display a progress bar while computing the covariance matrix. Default: False.

    Returns
    -------
    np.array
        Computed covariance matrix.

    """    
    Ncross = Cls_cmb[0].shape[0]
    Nfreqs = int((np.sqrt(1 + 8*Ncross) - 1) / 2)
    Nbins = w[0].wsp.bin.n_bands
    doublets = band_doublet(Nfreqs)
    
    f0 = nmt.NmtField(mask, None, spin=0)
    cw0 = nmt.NmtCovarianceWorkspace()
    cw0.compute_coupling_coefficients(f0, f0)
    lmax = cw0.wsp.lmax
    
    f2 = nmt.NmtField(mask, None, spin=2)
    cw2 = nmt.NmtCovarianceWorkspace()
    cw2.compute_coupling_coefficients(f2, f2)
    
    covmat = np.zeros((3, Nbins*Ncross, Nbins*Ncross))
    
    if progress:
        pbar = tqdm(desc='Estimating covariance matrix', total=int(Ncross*(Ncross+1)/2))
    
    for crossAB in range(Ncross):
        for crossCD in range(crossAB, Ncross):
            A, B = doublets[crossAB]
            C, D = doublets[crossCD]
            
            crossAC = cross_index(A, C, Nfreqs)
            crossBD = cross_index(B, D, Nfreqs)
            crossAD = cross_index(A, D, Nfreqs)
            crossBC = cross_index(B, C, Nfreqs)
            
            bands = np.array([A, B, C, D])
            counter = Counter(bands)
            
            if A == C and B == D:
                if A == B:
                    TT_a1b1 = Cls_cmb[0][crossAC] + Cls_fg[0][crossAC] + 2*Nls[0][crossAC]
                    TT_a1b2 = Cls_cmb[0][crossAD] + Cls_fg[0][crossAD]
                    TT_a2b1 = Cls_cmb[0][crossBC] + Cls_fg[0][crossBC]
                    TT_a2b2 = Cls_cmb[0][crossBD] + Cls_fg[0][crossBD] + 2*Nls[0][crossBD]
                    
                    EE_a1b1 = Cls_cmb[1][crossAC] + Cls_fg[1][crossAC] + 2*Nls[1][crossAC]
                    EE_a1b2 = Cls_cmb[1][crossAD] + Cls_fg[1][crossAD]
                    EE_a2b1 = Cls_cmb[1][crossBC] + Cls_fg[1][crossBC]
                    EE_a2b2 = Cls_cmb[1][crossBD] + Cls_fg[1][crossBD] + 2*Nls[1][crossBD]
                    
                    BB_a1b1 = Cls_cmb[2][crossAC] + Cls_fg[2][crossAC] + 2*Nls[2][crossAC]
                    BB_a1b2 = Cls_cmb[2][crossAD] + Cls_fg[2][crossAD]
                    BB_a2b1 = Cls_cmb[2][crossBC] + Cls_fg[2][crossBC]
                    BB_a2b2 = Cls_cmb[2][crossBD] + Cls_fg[2][crossBD] + 2*Nls[2][crossBD]
                    
                else:
                    TT_a1b1 = Cls_cmb[0][crossAC] + Cls_fg[0][crossAC] + Nls[0][crossAC]
                    TT_a1b2 = Cls_cmb[0][crossAD] + Cls_fg[0][crossAD]
                    TT_a2b1 = Cls_cmb[0][crossBC] + Cls_fg[0][crossBC]
                    TT_a2b2 = Cls_cmb[0][crossBD] + Cls_fg[0][crossBD] + Nls[0][crossBD]
                    
                    EE_a1b1 = Cls_cmb[1][crossAC] + Cls_fg[1][crossAC] + Nls[1][crossAC]
                    EE_a1b2 = Cls_cmb[1][crossAD] + Cls_fg[1][crossAD]
                    EE_a2b1 = Cls_cmb[1][crossBC] + Cls_fg[1][crossBC]
                    EE_a2b2 = Cls_cmb[1][crossBD] + Cls_fg[1][crossBD] + Nls[1][crossBD]
                    
                    BB_a1b1 = Cls_cmb[2][crossAC] + Cls_fg[2][crossAC] + Nls[2][crossAC]
                    BB_a1b2 = Cls_cmb[2][crossAD] + Cls_fg[2][crossAD]
                    BB_a2b1 = Cls_cmb[2][crossBC] + Cls_fg[2][crossBC]
                    BB_a2b2 = Cls_cmb[2][crossBD] + Cls_fg[2][crossBD] + Nls[2][crossBD]
                    
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
                
                TT_a1b1 = Cls_cmb[0][crossAC] + Cls_fg[0][crossAC] + Nls[0][crossAC]
                TT_a1b2 = Cls_cmb[0][crossAD] + Cls_fg[0][crossAD]
                TT_a2b1 = Cls_cmb[0][crossBC] + Cls_fg[0][crossBC]
                TT_a2b2 = Cls_cmb[0][crossBD] + Cls_fg[0][crossBD]
                
                EE_a1b1 = Cls_cmb[1][crossAC] + Cls_fg[1][crossAC] + Nls[1][crossAC]
                EE_a1b2 = Cls_cmb[1][crossAD] + Cls_fg[1][crossAD]
                EE_a2b1 = Cls_cmb[1][crossBC] + Cls_fg[1][crossBC]
                EE_a2b2 = Cls_cmb[1][crossBD] + Cls_fg[1][crossBD]
                
                BB_a1b1 = Cls_cmb[2][crossAC] + Cls_fg[2][crossAC] + Nls[2][crossAC]
                BB_a1b2 = Cls_cmb[2][crossAD] + Cls_fg[2][crossAD]
                BB_a2b1 = Cls_cmb[2][crossBC] + Cls_fg[2][crossBC]
                BB_a2b2 = Cls_cmb[2][crossBD] + Cls_fg[2][crossBD]
                
            else:
                TT_a1b1 = Cls_cmb[0][crossAC] + Cls_fg[0][crossAC]
                TT_a1b2 = Cls_cmb[0][crossAD] + Cls_fg[0][crossAD]
                TT_a2b1 = Cls_cmb[0][crossBC] + Cls_fg[0][crossBC]
                TT_a2b2 = Cls_cmb[0][crossBD] + Cls_fg[0][crossBD]
                
                EE_a1b1 = Cls_cmb[1][crossAC] + Cls_fg[1][crossAC]
                EE_a1b2 = Cls_cmb[1][crossAD] + Cls_fg[1][crossAD]
                EE_a2b1 = Cls_cmb[1][crossBC] + Cls_fg[1][crossBC]
                EE_a2b2 = Cls_cmb[1][crossBD] + Cls_fg[1][crossBD]
                
                BB_a1b1 = Cls_cmb[2][crossAC] + Cls_fg[2][crossAC]
                BB_a1b2 = Cls_cmb[2][crossAD] + Cls_fg[2][crossAD]
                BB_a2b1 = Cls_cmb[2][crossBC] + Cls_fg[2][crossBC]
                BB_a2b2 = Cls_cmb[2][crossBD] + Cls_fg[2][crossBD]
                
            Cl0_a1b1 = [TT_a1b1]
            Cl0_a1b2 = [TT_a1b2]
            Cl0_a2b1 = [TT_a2b1]
            Cl0_a2b2 = [TT_a2b2]
                
            Cl2_a1b1 = [EE_a1b1, np.zeros(lmax+1), np.zeros(lmax+1), BB_a1b1]
            Cl2_a1b2 = [EE_a1b2, np.zeros(lmax+1), np.zeros(lmax+1), BB_a1b2]
            Cl2_a2b1 = [EE_a2b1, np.zeros(lmax+1), np.zeros(lmax+1), BB_a2b1]
            Cl2_a2b2 = [EE_a2b2, np.zeros(lmax+1), np.zeros(lmax+1), BB_a2b2]
            
            if output in ['TT', 'all']:
                covT = nmt.gaussian_covariance(cw0, 0, 0, 0, 0, Cl0_a1b1, Cl0_a1b2, Cl0_a2b1, Cl0_a2b2, w[0]).reshape([Nbins, 1, Nbins, 1])
            if output in ['EE', 'all']:
                covE = nmt.gaussian_covariance(cw2, 2, 2, 2, 2, Cl2_a1b1, Cl2_a1b2, Cl2_a2b1, Cl2_a2b2, w[1]).reshape([Nbins, 4, Nbins, 4])
            if output in ['BB', 'all']:
                covB = nmt.gaussian_covariance(cw2, 2, 2, 2, 2, Cl2_a1b1, Cl2_a1b2, Cl2_a2b1, Cl2_a2b2, w[2]).reshape([Nbins, 4, Nbins, 4])
            
            if corfg:
                TT_a1b1 = Cls_fg[0][crossAC]
                TT_a1b2 = Cls_fg[0][crossAD]
                TT_a2b1 = Cls_fg[0][crossBC]
                TT_a2b2 = Cls_fg[0][crossBD]
                
                EE_a1b1 = Cls_fg[1][crossAC]
                EE_a1b2 = Cls_fg[1][crossAD]
                EE_a2b1 = Cls_fg[1][crossBC]
                EE_a2b2 = Cls_fg[1][crossBD]
                
                BB_a1b1 = Cls_fg[2][crossAC]
                BB_a1b2 = Cls_fg[2][crossAD]
                BB_a2b1 = Cls_fg[2][crossBC]
                BB_a2b2 = Cls_fg[2][crossBD]
                
                Cl0_a1b1 = [TT_a1b1]
                Cl0_a1b2 = [TT_a1b2]
                Cl0_a2b1 = [TT_a2b1]
                Cl0_a2b2 = [TT_a2b2]
                    
                Cl2_a1b1 = [EE_a1b1, np.zeros(lmax+1), np.zeros(lmax+1), BB_a1b1]
                Cl2_a1b2 = [EE_a1b2, np.zeros(lmax+1), np.zeros(lmax+1), BB_a1b2]
                Cl2_a2b1 = [EE_a2b1, np.zeros(lmax+1), np.zeros(lmax+1), BB_a2b1]
                Cl2_a2b2 = [EE_a2b2, np.zeros(lmax+1), np.zeros(lmax+1), BB_a2b2]
                
                if output in ['TT', 'all']:
                    covT -= nmt.gaussian_covariance(cw0, 0, 0, 0, 0, Cl0_a1b1, Cl0_a1b2, Cl0_a2b1, Cl0_a2b2, w[0]).reshape([Nbins, 1, Nbins, 1])
                if output in ['EE', 'all']:
                    covE -= nmt.gaussian_covariance(cw2, 2, 2, 2, 2, Cl2_a1b1, Cl2_a1b2, Cl2_a2b1, Cl2_a2b2, w[1]).reshape([Nbins, 4, Nbins, 4])
                if output in ['BB', 'all']:
                    covB -= nmt.gaussian_covariance(cw2, 2, 2, 2, 2, Cl2_a1b1, Cl2_a1b2, Cl2_a2b1, Cl2_a2b2, w[2]).reshape([Nbins, 4, Nbins, 4])
            
            for i in range(Nbins):
                for j in range(Nbins):
                    if output in ['TT', 'all']:
                        covmat[0, i*Ncross + crossAB, j*Ncross + crossCD] = covT[i, 0, j, 0]
                        
                        if crossAB != crossCD:
                            covmat[0, i*Ncross + crossCD, j*Ncross + crossAB] = covT[i, 0, j, 0]
                        
                    if output in ['EE', 'all']:
                        covmat[1, i*Ncross + crossAB, j*Ncross + crossCD] = covE[i, 0, j, 0]
                        
                        if crossAB != crossCD:
                            covmat[1, i*Ncross + crossCD, j*Ncross + crossAB] = covE[i, 0, j, 0]
                        
                    if output in ['BB', 'all']:
                        covmat[2, i*Ncross + crossAB, j*Ncross + crossCD] = covB[i, 3, j, 3]
                        
                        if crossAB != crossCD:
                            covmat[2, i*Ncross + crossCD, j*Ncross + crossAB] = covB[i, 3, j, 3]
                    
            if progress:
                pbar.update(1)
                
    if progress:
        pbar.close()
    
    if output == 'TT':
        return covmat[0]
        
    elif output == 'EE':
        return covmat[1]
    
    elif output == 'BB':
        return covmat[2]
        
    return covmat

def cov_NaMaster_signal(mask, Cls, w, corfg=True, output='all', progress=False):
    """
    Compute analytical covariance matrix using NaMaster fonctions directly from signal.
    Cls can be mode-decoupled unbinned power spectra, but better accuracy is achieved if they are mode-coupled pseudo-Cls divided by fsky.

    Parameters
    ----------
    mask : np.array
        Mask used for computing the power spectra.
    Cls : list
        Signal cross-power spectra. Must contain 3 numpy arrays of dimension (Ncross, 3*NSIDE) corresponding to TT, EE and BB spectra.
        w : list
            List of nmt.NmtWorkspace used for computing TT, EE and BB power spectra.
    corfg : bool, optional
        If True, correct for the cosmic variance of foregrounds. Default: True.
    output : string, optional
        If 'TT', return covariance matrix for temperature.
        If 'EE', return covariance matrix for E-modes.
        If 'BB', return covariance matrix for B-modes.
        If 'all', return covariance matrices of TT, EE, BB spectra.
        Default: 'all'.
    progress : bool, optional
        If True, display a progress bar while computing the covariance matrix. Default: False.

    Returns
    -------
    np.array
        Computed covariance matrix.

    """
    Ncross = Cls[0].shape[0]
    Nfreqs = int((np.sqrt(1 + 8*Ncross) - 1) / 2)
    Nbins = w[0].wsp.bin.n_bands
    doublets = band_doublet(Nfreqs)
    
    f0 = nmt.NmtField(mask, None, spin=0)
    cw0 = nmt.NmtCovarianceWorkspace()
    cw0.compute_coupling_coefficients(f0, f0)
    lmax = cw0.wsp.lmax
    
    f2 = nmt.NmtField(mask, None, spin=2)
    cw2 = nmt.NmtCovarianceWorkspace()
    cw2.compute_coupling_coefficients(f2, f2)
    
    covmat = np.zeros((3, Nbins*Ncross, Nbins*Ncross))
    
    if progress:
        pbar = tqdm(desc='Estimating covariance matrix', total=int(Ncross*(Ncross+1)/2))
    
    for crossAB in range(Ncross):
        for crossCD in range(crossAB, Ncross):
            A, B = doublets[crossAB]
            C, D = doublets[crossCD]
            
            crossAC = cross_index(A, C, Nfreqs)
            crossBD = cross_index(B, D, Nfreqs)
            crossAD = cross_index(A, D, Nfreqs)
            crossBC = cross_index(B, C, Nfreqs)
            
            TT_a1b1 = Cls[0][crossAC]
            TT_a1b2 = Cls[0][crossAD]
            TT_a2b1 = Cls[0][crossBC]
            TT_a2b2 = Cls[0][crossBD]
            
            EE_a1b1 = Cls[1][crossAC]
            EE_a1b2 = Cls[1][crossAD]
            EE_a2b1 = Cls[1][crossBC]
            EE_a2b2 = Cls[1][crossBD]
            
            BB_a1b1 = Cls[2][crossAC]
            BB_a1b2 = Cls[2][crossAD]
            BB_a2b1 = Cls[2][crossBC]
            BB_a2b2 = Cls[2][crossBD]
            
            Cl0_a1b1 = [TT_a1b1]
            Cl0_a1b2 = [TT_a1b2]
            Cl0_a2b1 = [TT_a2b1]
            Cl0_a2b2 = [TT_a2b2]
            
            Cl2_a1b1 = [EE_a1b1, np.zeros(lmax+1), np.zeros(lmax+1), BB_a1b1]
            Cl2_a1b2 = [EE_a1b2, np.zeros(lmax+1), np.zeros(lmax+1), BB_a1b2]
            Cl2_a2b1 = [EE_a2b1, np.zeros(lmax+1), np.zeros(lmax+1), BB_a2b1]
            Cl2_a2b2 = [EE_a2b2, np.zeros(lmax+1), np.zeros(lmax+1), BB_a2b2]
            
            if output in ['TT', 'all']:
                covT = nmt.gaussian_covariance(cw0, 0, 0, 0, 0, Cl0_a1b1, Cl0_a1b2, Cl0_a2b1, Cl0_a2b2, w[0]).reshape([Nbins, 1, Nbins, 1])
            if output in ['EE', 'all']:
                covE = nmt.gaussian_covariance(cw2, 2, 2, 2, 2, Cl2_a1b1, Cl2_a1b2, Cl2_a2b1, Cl2_a2b2, w[1]).reshape([Nbins, 4, Nbins, 4])
            if output in ['BB', 'all']:
                covB = nmt.gaussian_covariance(cw2, 2, 2, 2, 2, Cl2_a1b1, Cl2_a1b2, Cl2_a2b1, Cl2_a2b2, w[2]).reshape([Nbins, 4, Nbins, 4])
            
            for i in range(Nbins):
                for j in range(Nbins):
                    if output in ['TT', 'all']:
                        covmat[0, i*Ncross + crossAB, j*Ncross + crossCD] = covT[i, 0, j, 0]
                        
                        if crossAB != crossCD:
                            covmat[0, i*Ncross + crossCD, j*Ncross + crossAB] = covT[i, 0, j, 0]
                        
                    if output in ['EE', 'all']:
                        covmat[1, i*Ncross + crossAB, j*Ncross + crossCD] = covE[i, 0, j, 0]
                        
                        if crossAB != crossCD:
                            covmat[1, i*Ncross + crossCD, j*Ncross + crossAB] = covE[i, 0, j, 0]
                        
                    if output in ['BB', 'all']:
                        covmat[2, i*Ncross + crossAB, j*Ncross + crossCD] = covB[i, 3, j, 3]
                        
                        if crossAB != crossCD:
                            covmat[2, i*Ncross + crossCD, j*Ncross + crossAB] = covB[i, 3, j, 3]
                    
            if progress:
                pbar.update(1)
                
    if progress:
        pbar.close()
        
    if progress:
        pbar.close()
    
    if output == 'TT':
        return covmat[0]
        
    elif output == 'EE':
        return covmat[1]
    
    elif output == 'BB':
        return covmat[2]
        
    return covmat

def compute_covmat(mask, w, Cls_signal=None, Cls_cmb=None, Cls_fg=None, Nls=None, type='Nmt-fg', output='all', progress=False):
    """
    Compute analytical covariance matrix in different ways.
    Cls can be mode-decoupled power spectra, but better accuracy is achieved if they are mode-coupled pseudo-Cls divided by fsky.
    Cls must be unbinned, except for 'Knox_signal' estimate for which they can be binned.

    Parameters
    ----------
    mask : np.array
        Mask used for computing the power spectra.
    w : list
        List of nmt.NmtWorkspace used for computing the power spectra.
        Must contain one nmt.NmtWorkspace for Knox estimates. For NaMaster estimates, must contain 3 workspaces corresponding to TT, EE and BB spectra.
    Cls_signal : list, optional
        Signal power spectra. Needed only if type is 'Knox_signal' or 'Nmt_signal'.
        Must contain 3 numpy arrays of dimension (Ncross, 3*NSIDE) corresponding to TT, EE and BB spectra, except for type 'Knox_signal' for which Cls can be binned.
        Default: None.
    Cls_cmb : list, optional
        CMB power spectra. Needed only for types other than 'Knox_signal' and 'Nmt_signal'.
        Must contain 3 numpy arrays of dimension (Ncross, 3*NSIDE) corresponding to TT, EE and BB spectra, except for type 'Knox_signal' for which Cls can be binned.
        Default: None.
    Cls_fg : list, optional
        Foregrounds power spectra. Needed only for types other than 'Knox_signal' and 'Nmt_signal'.
        Must contain 3 numpy arrays of dimension (Ncross, 3*NSIDE) corresponding to TT, EE and BB spectra, except for type 'Knox_signal' for which Cls can be binned.
        Default: None.
    Nls : np.array, optional
        Noise power spectra. Needed only for types other than 'Knox_signal' and 'Nmt_signal'.
        Must contain 3 numpy arrays of dimension (Ncross, 3*NSIDE) corresponding to TT, EE and BB spectra, except for type 'Knox_signal' for which Cls can be binned.
        Default: None.
    type : string, optional
        Type of the estimate. Can be 'Knox-fg', 'Knox+fg', 'Knox_signal', 'Nmt-fg', 'Nmt+fg', 'Nmt_signal'.
        Default: 'Nmt-fg'.
    output : string, optional
        If 'TT', return covariance matrix for temperature.
        If 'EE', return covariance matrix for E-modes.
        If 'BB', return covariance matrix for B-modes.
        If 'all', return covariance matrices of TT, EE, BB spectra.
        For Knox estimates, must be 'EE' or 'BB'. Default: 'all'.
    progress : bool, optional
        If True, display a progress bar while computing the covariance matrix. Default: False.

    Returns
    -------
    np.array
        Computed covariance matrix.

    """
    lmax, Nbins = w[0].wsp.lmax, w[0].wsp.bin.n_bands
    
    delta_l = int(lmax / Nbins)
    b = nmt.NmtBin.from_lmax_linear(lmax, nlb=delta_l, is_Dell=True)

    if type != 'Knox_signal':
        if type == 'Nmt_signal':
            for i in range(3):
                if Cls_signal[i].shape[1] == Nbins:
                    raise ValueError("Cls must not be binned for types other than 'Knox_signal'")

        else:
            for i in range(3):
                if Cls_cmb[i].shape[1] == Nbins or Cls_fg[i].shape[1] == Nbins or Nls[i].shape[1] == Nbins:
                    raise ValueError("Cls must not be binned for types other than 'Knox_signal'")

    if type in ['Knox-fg', 'Knox+fg']:
        if output == 'TT':
            w = w[0]
            Cls_cmb_binned = np.zeros((Cls_cmb[0].shape[0], Nbins))
            Cls_fg_binned = np.zeros((Cls_fg[0].shape[0], Nbins))
            Nls_binned = np.zeros((Nls[0].shape[0], Nbins))
            
            for i in range(Cls_cmb[0].shape[0]):
                Cls_cmb_binned[i] = b.bin_cell(Cls_cmb[0][i, :lmax+1])
                Cls_fg_binned[i] = b.bin_cell(Cls_fg[0][i, :lmax+1])
                Nls_binned[i] = b.bin_cell(Nls[0][i, :lmax+1])
        
        elif output == 'EE':
            w = w[1]
            Cls_cmb_binned = np.zeros((Cls_cmb[1].shape[0], Nbins))
            Cls_fg_binned = np.zeros((Cls_fg[1].shape[0], Nbins))
            Nls_binned = np.zeros((Nls[1].shape[0], Nbins))
            
            for i in range(Cls_cmb[1].shape[0]):
                Cls_cmb_binned[i] = b.bin_cell(Cls_cmb[1][i, :lmax+1])
                Cls_fg_binned[i] = b.bin_cell(Cls_fg[1][i, :lmax+1])
                Nls_binned[i] = b.bin_cell(Nls[1][i, :lmax+1])
            
        elif output == 'BB':
            w = w[2]
            Cls_cmb_binned = np.zeros((Cls_cmb[2].shape[0], Nbins))
            Cls_fg_binned = np.zeros((Cls_fg[2].shape[0], Nbins))
            Nls_binned = np.zeros((Nls[2].shape[0], Nbins))
            
            for i in range(Cls_cmb[2].shape[0]):
                Cls_cmb_binned[i] = b.bin_cell(Cls_cmb[2][i, :lmax+1])
                Cls_fg_binned[i] = b.bin_cell(Cls_fg[2][i, :lmax+1])
                Nls_binned[i] = b.bin_cell(Nls[2][i, :lmax+1])
            
        else:
            raise ValueError("Incorrect type for output 'all'")
            
        if type == 'Knox-fg':
            return cov_Knox(mask, Cls_cmb_binned, Cls_fg_binned, Nls_binned, w, corfg=True, progress=progress)
        
        elif type == 'Knox+fg':
            return cov_Knox(mask, Cls_cmb_binned, Cls_fg_binned, Nls_binned, w, corfg=False, progress=progress)
        
    elif type == 'Knox_signal':
        if output == 'TT':
            Cls_signal = Cls_signal[0]
        
        if output == 'EE':
            Cls_signal = Cls_signal[1]
            
        elif output == 'BB':
            Cls_signal = Cls_signal[2]
            
        else:
            raise ValueError("Incorrect type for output 'all'")
            
        if Cls_signal.shape[1] > Nbins:
            Cls_signal_binned = np.zeros((Cls_signal.shape[0], Nbins))
            
            for i in range(Cls_signal.shape[0]):
                Cls_signal_binned[i] = b.bin_cell(Cls_signal[i, :lmax+1])
                
            Cls_signal = Cls_signal_binned
            
        return cov_Knox_signal(mask, Cls_signal, w, progress=progress)
            
    if type == 'Nmt-fg':
        return cov_NaMaster(mask, Cls_cmb, Cls_fg, Nls, w, corfg=True, output=output, progress=progress)
        
    elif type == 'Nmt+fg':
        return cov_NaMaster(mask, Cls_cmb, Cls_fg, Nls, w, corfg=False, output=output, progress=progress)
            
    elif type == 'Nmt_signal':
        return cov_NaMaster_signal(mask, Cls_signal, w, output=output, progress=progress)
            
    else:
        raise ValueError('Unknown type of estimate')

def inverse_covmat(covmat, Ncross, neglect_corbins=False, return_cholesky=False, return_new=False):
    """
    Inverse the input covariance matrix to the best achievable accuracy.

    Parameters
    ----------
    covmat : np.array
        Covariance matrix to inverse. Must be of dimension (Ncross*Nbins, Ncross*Nbins).
    Ncross : int
        Number of cross-spectra the covariance matrix has been computed from.
    neglect_corbins : bool, optional
        If True, neglect correlations between ell bins. Default: False.
    return_cholesky : bool, optional
        If True, return the Cholesky matrix computed from the inverse covariance, otherwise return the standard inverse matrix. Default: False.
    return_new : bool, optional
        If True, return also the altered covariance matrix such that the computed inverse is positive definite. Default: False.

    Returns
    -------
    np.array
        Inversed covariance matrix.

    """
    if neglect_corbins:
        Nbins = int(len(covmat) / Ncross)
        
        inv_covmat = np.zeros((Nbins, Ncross, Ncross))
        new_covmat = np.zeros((Nbins, Ncross, Ncross))
        
        for i in range(Nbins):
            inv_covmat[i], new_covmat[i] = compute_inverse(covmat[i*Ncross:(i+1)*Ncross, i*Ncross:(i+1)*Ncross])
            
            if return_cholesky:
                inv_covmat[i] = np.linalg.cholesky(inv_covmat[i])
            
    else:
        inv_covmat, new_covmat = compute_inverse(covmat)
        
        if return_cholesky:
            inv_covmat = np.linalg.cholesky(inv_covmat)
        
    if return_new:
        return inv_covmat, new_covmat
    else:
        return inv_covmat
