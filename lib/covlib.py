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


def band_doublet(index, Nf):
    """
    Find the band doublet associated to the input cross-spectrum index.

    Parameters
    ----------
    index : int
        Index of the cross-spectrum.
    Nf : int
        Number of frequency bands.

    Returns
    -------
    np.array
        Band doublet associated to the cross-spectrum index.

    """
    for i in range(Nf):
        for j in range(i, Nf):
            try:
                cross = np.vstack((cross, np.array([i, j])))
            except:
                cross = np.array([i, j])
                
    return cross[index]

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
    
    b = nmt.NmtBin.from_lmax_linear(lmax, nlb=delta_l)
    leff = b.get_effective_ells()
    
    nu_l = (2*leff+1) * delta_l * np.mean(mask**2)**2 / np.mean(mask**4)
    
    covmat = np.zeros((Nbins*Ncross, Nbins*Ncross))
    
    if progress:
        pbar = tqdm(desc='Estimating covariance matrix', total=int(Ncross*(Ncross+1)/2))
    
    for crossAB in range(Ncross):
        for crossCD in range(crossAB, Ncross):
            A, B = band_doublet(crossAB, Nfreqs)
            C, D = band_doublet(crossCD, Nfreqs)
            
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
    
    b = nmt.NmtBin.from_lmax_linear(lmax, nlb=delta_l)
    leff = b.get_effective_ells()
    
    nu_l = (2*leff+1) * delta_l * np.mean(mask**2)**2 / np.mean(mask**4)
    
    covmat = np.zeros((Nbins*Ncross, Nbins*Ncross))
    
    if progress:
        pbar = tqdm(desc='Estimating covariance matrix', total=int(Ncross*(Ncross+1)/2))
    
    for crossAB in range(Ncross):
        for crossCD in range(crossAB, Ncross):
            A, B = band_doublet(crossAB, Nfreqs)
            C, D = band_doublet(crossCD, Nfreqs)
            
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

def cov_NaMaster(mask, Cls_cmb_EE, Cls_cmb_BB, Cls_fg_EE, Cls_fg_BB, Nls_EE, Nls_BB, w, corfg=True, output='all', progress=False):
    """
    Compute analytical covariance matrix using NaMaster fonctions from theoretical power spectra and noise model.
    Cls can be mode-decoupled unbinned power spectra, but better accuracy is achieved if they are mode-coupled pseudo-Cls divided by fsky.

    Parameters
    ----------
    mask : np.array
        Mask used for computing the power spectra.
    Cls_cmb_EE : np.array
        CMB EE power spectra. Must be of dimension (Ncross, 3*NSIDE).
    Cls_cmb_BB : np.array
        CMB BB power spectra. Must be of dimension (Ncross, 3*NSIDE).
    Cls_fg_EE : np.array
        Foregrounds EE power spectra. Must be of dimension (Ncross, 3*NSIDE).
    Cls_fg_BB : np.array
        Foregrounds BB power spectra. Must be of dimension (Ncross, 3*NSIDE).
    Nls_EE : np.array
        Noise EE power spectra. Must be of dimension (Ncross, 3*NSIDE).
    Nls_BB : np.array
        Noise BB power spectra. Must be of dimension (Ncross, 3*NSIDE).
    w : nmt.NmtWorkspace
        Workspace used for computing the power spectra.
    corfg : bool, optional
        If True, correct for the cosmic variance of foregrounds. Default: True.
    output : string, optional
        If 'EE', compute covariance matrix for E modes only. If 'BB', compute covariance matrix for B modes only. If 'all', compute the full covariance matrix. Default: 'all'.
    progress : bool, optional
        If True, display a progress bar while computing the covariance matrix. Default: False.

    Returns
    -------
    np.array
        Computed covariance matrix.

    """
    Ncross = Cls_cmb_EE.shape[0]
    Nfreqs = int((np.sqrt(1 + 8*Ncross) - 1) / 2)
    Nbins = w.wsp.bin.n_bands
    
    f = nmt.NmtField(mask, None, spin=2)
    cw = nmt.NmtCovarianceWorkspace()
    cw.compute_coupling_coefficients(f, f)
    lmax = cw.wsp.lmax
    
    if output == 'all':
        covmat = np.zeros((4*Nbins*Ncross, 4*Nbins*Ncross))
    else:
        covmat = np.zeros((Nbins*Ncross, Nbins*Ncross))
    
    if progress:
        pbar = tqdm(desc='Estimating covariance matrix', total=int(Ncross*(Ncross+1)/2))
    
    for crossAB in range(Ncross):
        for crossCD in range(crossAB, Ncross):
            A, B = band_doublet(crossAB, Nfreqs)
            C, D = band_doublet(crossCD, Nfreqs)
            
            crossAC = cross_index(A, C, Nfreqs)
            crossBD = cross_index(B, D, Nfreqs)
            crossAD = cross_index(A, D, Nfreqs)
            crossBC = cross_index(B, C, Nfreqs)
            
            bands = np.array([A, B, C, D])
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
            
            cov = nmt.gaussian_covariance(cw, 2, 2, 2, 2, Cl_a1b1, Cl_a1b2, Cl_a2b1, Cl_a2b2, w)
            
            if corfg:
                EE_a1b1 = Cls_fg_EE[crossAC]
                EE_a1b2 = Cls_fg_EE[crossAD]
                EE_a2b1 = Cls_fg_EE[crossBC]
                EE_a2b2 = Cls_fg_EE[crossBD]
                
                BB_a1b1 = Cls_fg_BB[crossAC]
                BB_a1b2 = Cls_fg_BB[crossAD]
                BB_a2b1 = Cls_fg_BB[crossBC]
                BB_a2b2 = Cls_fg_BB[crossBD]
                
                Cl_a1b1 = [EE_a1b1, np.zeros(lmax+1), np.zeros(lmax+1), BB_a1b1]
                Cl_a1b2 = [EE_a1b2, np.zeros(lmax+1), np.zeros(lmax+1), BB_a1b2]
                Cl_a2b1 = [EE_a2b1, np.zeros(lmax+1), np.zeros(lmax+1), BB_a2b1]
                Cl_a2b2 = [EE_a2b2, np.zeros(lmax+1), np.zeros(lmax+1), BB_a2b2]
                
                cov -= nmt.gaussian_covariance(cw, 2, 2, 2, 2, Cl_a1b1, Cl_a1b2, Cl_a2b1, Cl_a2b2, w)
            
            cov = cov.reshape([Nbins, 4, Nbins, 4])
            
            if output == 'EE':
                for i in range(Nbins):
                    for j in range(Nbins):
                        covmat[i*Ncross + crossAB, j*Ncross + crossCD] = cov[i, 0, j, 0]
                        
                        if crossAB != crossCD:
                            covmat[i*Ncross + crossCD, j*Ncross + crossAB] = cov[i, 0, j, 0]
            
            elif output == 'BB':
                for i in range(Nbins):
                    for j in range(Nbins):
                        covmat[i*Ncross + crossAB, j*Ncross + crossCD] = cov[i, 3, j, 3]
                        
                        if crossAB != crossCD:
                            covmat[i*Ncross + crossCD, j*Ncross + crossAB] = cov[i, 3, j, 3]
            
            else:
                for WX in range(4):
                    for YZ in range(4):
                        for i in range(Nbins):
                            for j in range(Nbins):
                                covmat[(WX*Nbins + i) * Ncross + crossAB, (YZ*Nbins + j) * Ncross + crossCD] = cov[i, WX, j, YZ]
                                
                                if crossAB != crossCD:
                                    covmat[(WX*Nbins + i) * Ncross + crossCD, (YZ*Nbins + j) * Ncross + crossAB] = cov[i, WX, j, YZ]
                    
            if progress:
                pbar.update(1)
                
    if progress:
        pbar.close()
        
    return covmat

def cov_NaMaster_signal(mask, Cls_EE, Cls_BB, w, corfg=True, output='all', progress=False):
    """
    Compute analytical covariance matrix using NaMaster fonctions directly from signal.
    Cls can be mode-decoupled unbinned power spectra, but better accuracy is achieved if they are mode-coupled pseudo-Cls divided by fsky.

    Parameters
    ----------
    mask : np.array
        Mask used for computing the power spectra.
    Cls_EE : np.array
        EE cross-power spectra. Must be of dimension (Ncross, 3*NSIDE).
    Cls_BB : np.array
        BB cross-power spectra. Must be of dimension (Ncross, 3*NSIDE).
    w : nmt.NmtWorkspace
        Workspace used for computing the power spectra.
    corfg : bool, optional
        If True, correct for the cosmic variance of foregrounds. Default: True.
    output : string, optional
        If 'EE', compute covariance matrix for E modes only. If 'BB', compute covariance matrix for B modes only. If 'all', compute the full covariance matrix. Default: 'all'.
    progress : bool, optional
        If True, display a progress bar while computing the covariance matrix. Default: False.

    Returns
    -------
    np.array
        Computed covariance matrix.

    """
    Ncross = Cls_EE.shape[0]
    Nfreqs = int((np.sqrt(1 + 8*Ncross) - 1) / 2)
    Nbins = w.wsp.bin.n_bands
    
    f = nmt.NmtField(mask, None, spin=2)
    cw = nmt.NmtCovarianceWorkspace()
    cw.compute_coupling_coefficients(f, f)
    lmax = cw.wsp.lmax
    
    if output == 'all':
        covmat = np.zeros((4*Nbins*Ncross, 4*Nbins*Ncross))
    else:
        covmat = np.zeros((Nbins*Ncross, Nbins*Ncross))
    
    if progress:
        pbar = tqdm(desc='Estimating covariance matrix', total=int(Ncross*(Ncross+1)/2))
    
    for crossAB in range(Ncross):
        for crossCD in range(crossAB, Ncross):
            A, B = band_doublet(crossAB, Nfreqs)
            C, D = band_doublet(crossCD, Nfreqs)
            
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
            
            cov = nmt.gaussian_covariance(cw, 2, 2, 2, 2, Cl_a1b1, Cl_a1b2, Cl_a2b1, Cl_a2b2, w)
            cov = cov.reshape([Nbins, 4, Nbins, 4])
            
            if output == 'EE':
                for i in range(Nbins):
                    for j in range(Nbins):
                        covmat[i*Ncross + crossAB, j*Ncross + crossCD] = cov[i, 0, j, 0]
                        
                        if crossAB != crossCD:
                            covmat[i*Ncross + crossCD, j*Ncross + crossAB] = cov[i, 0, j, 0]
            
            elif output == 'BB':
                for i in range(Nbins):
                    for j in range(Nbins):
                        covmat[i*Ncross + crossAB, j*Ncross + crossCD] = cov[i, 3, j, 3]
                        
                        if crossAB != crossCD:
                            covmat[i*Ncross + crossCD, j*Ncross + crossAB] = cov[i, 3, j, 3]
            
            else:
                for WX in range(4):
                    for YZ in range(4):
                        for i in range(Nbins):
                            for j in range(Nbins):
                                covmat[(WX*Nbins + i) * Ncross + crossAB, (YZ*Nbins + j) * Ncross + crossCD] = cov[i, WX, j, YZ]
                                
                                if crossAB != crossCD:
                                    covmat[(WX*Nbins + i) * Ncross + crossCD, (YZ*Nbins + j) * Ncross + crossAB] = cov[i, WX, j, YZ]
                    
            if progress:
                pbar.update(1)
                
    if progress:
        pbar.close()
        
    return covmat

def compute_covmat(mask, w, Cls_signal_EE=None, Cls_signal_BB=None, Cls_cmb_EE=None, Cls_cmb_BB=None, Cls_fg_EE=None, Cls_fg_BB=None, Nls_EE=None, Nls_BB=None, type='Nmt-fg', output='all', progress=False):
    """
    Compute analytical covariance matrix in different ways.
    Cls can be mode-decoupled power spectra, but better accuracy is achieved if they are mode-coupled pseudo-Cls divided by fsky.
    Cls must be binned for Knox estimates, but not for NaMaster estimates.

    Parameters
    ----------
    mask : np.array
        Mask used for computing the power spectra.
    w : nmt.NmtWorkspace
        Workspace used for computing the power spectra.
    Cls_signal_EE : np.array, optional
        Signal EE power spectra. Needed only if type is 'Knox_signal' or 'Nmt_signal'. Default: None.
    Cls_signal_BB : np.array, optional
        Signal BB power spectra. Needed only if type is 'Knox_signal' or 'Nmt_signal'. Default: None.
    Cls_cmb_EE : np.array, optional
        CMB EE power spectra. Needed only for types other than 'Knox_signal' and 'Nmt_signal'. Default: None.
    Cls_cmb_BB : np.array, optional
        CMB BB power spectra. Needed only for types other than 'Knox_signal' and 'Nmt_signal'. Default: None.
    Cls_fg_EE : np.array, optional
        Foregrounds EE power spectra. Needed only for types other than 'Knox_signal' and 'Nmt_signal'. Default: None.
    Cls_fg_BB : np.array, optional
        Foregrounds BB power spectra. Needed only for types other than 'Knox_signal' and 'Nmt_signal'. Default: None.
    Nls_EE : np.array, optional
        Noise EE power spectra. Needed only for types other than 'Knox_signal' and 'Nmt_signal'. Default: None.
    Nls_BB : np.array, optional
        Noise BB power spectra. Needed only for types other than 'Knox_signal' and 'Nmt_signal'. Default: None.
    type : string, optional
        Type of the estimate. Can be 'Knox-fg', 'Knox+fg', 'Knox_signal', 'Nmt-fg', 'Nmt+fg', 'Nmt_signal'.
        Default: 'Nmt-fg'.
    output : string, optional
        If 'EE', compute covariance matrix for E modes only. If 'BB', compute covariance matrix for B modes only. If 'all', compute the full covariance matrix. For Knox estimates, must be 'EE' or 'BB'. Default: 'all'.
    progress : bool, optional
        If True, display a progress bar while computing the covariance matrix. Default: False.

    Returns
    -------
    np.array
        Computed covariance matrix.

    """
    if type in ['Knox-fg', 'Knox+fg', 'Knox_signal']:
        if output == 'EE':
            Cls_signal = Cls_signal_EE
            Cls_cmb = Cls_cmb_EE
            Cls_fg = Cls_fg_EE
            Nls = Nls_EE
            
        elif output == 'BB':
            Cls_signal = Cls_signal_BB
            Cls_cmb = Cls_cmb_BB
            Cls_fg = Cls_fg_BB
            Nls = Nls_BB
            
        else:
            raise ValueError("Incorrect type for output 'all'")
            
        if type == 'Knox-fg':
            return cov_Knox(mask, Cls_cmb, Cls_fg, Nls, w, corfg=True, progress=progress)
        
        elif type == 'Knox+fg':
            return cov_Knox(mask, Cls_cmb, Cls_fg, Nls, w, corfg=False, progress=progress)
            
        elif type == 'Knox_signal':
            return cov_Knox_signal(mask, Cls_signal, w, progress=progress)
            
    else:
        if type == 'Nmt-fg':
            return cov_NaMaster(mask, Cls_cmb_EE, Cls_cmb_BB, Cls_fg_EE, Cls_fg_BB, Nls_EE, Nls_BB, w, corfg=True, output=output, progress=progress)
        
        elif type == 'Nmt+fg':
            return cov_NaMaster(mask, Cls_cmb_EE, Cls_cmb_BB, Cls_fg_EE, Cls_fg_BB, Nls_EE, Nls_BB, w, corfg=False, output=output, progress=progress)
            
        elif type == 'Nmt_signal':
            return cov_NaMaster_signal(mask, Cls_signal_EE, Cls_signal_BB, w, output=output, progress=progress)
            
        else:
            raise ValueError("Unknown type of estimate")

def inverse_covmat(covmat, Nspec, neglect_corbins=True, return_cholesky=False, return_new=False):
    """
    Inverse the input covariance matrix to the best achievable accuracy.

    Parameters
    ----------
    covmat : np.array
        Covariance matrix to inverse. Must be of dimension (Ncross*Nbins, Ncross*Nbins).
    Nspec : int
        Number of spectra the covariance matrix has been computed from. Should be 4*Ncross if all E and B modes were kept, otherwise should be Ncross.
    neglect_corbins : bool, optional
        If True, neglect correlations between ell bins. Default: True.
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
        Nbins = int(len(covmat) / Nspec)
        
        inv_covmat = np.zeros((Nbins, Nspec, Nspec))
        new_covmat = np.zeros((Nbins, Nspec, Nspec))
        
        for i in range(Nbins):
            inv_covmat[i], new_covmat[i] = compute_inverse(covmat[i*Nspec:(i+1)*Nspec, i*Nspec:(i+1)*Nspec])
            
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