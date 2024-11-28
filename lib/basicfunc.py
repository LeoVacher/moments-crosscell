from astropy import constants as const
#import scipy.constants as constants
import numpy as np
import healpy as hp
import pysm3.units as u
import pysm_common as psm 
import sympy as sp

#FONCTIONS

def B(nu, T):
    """Planck function.

    :param nu: frequency in GHz at which to evaluate planck function.
    :type nu: float.
    :param T: temperature of black body in Kelvins.
    :type T: float.
    :return: float -- black body brightness.

    """
    x = const.h.value*nu*1.e9/const.k_B.value/T
    return 2.*const.h.value *(nu *1.e9)**3/ const.c.value**2/np.expm1(x)

def mbb(nu,beta,T):
    """Modified black body.

    :param nu: frequency in GHz at which to evaluate planck function.
    :type nu: float.
    :param beta: spectral index of the emissivity
    :type beta: float    
    :param T: temperature of black body in Kelvins.
    :type T: float.
    :return: float -- modified black body brightness.

    """    
    return B(nu,T)*(1e9*nu)**beta


def mbb_uK(nu,beta,T,nu0=353.):
    """Modified black body.

    :param nu: frequency in GHz at which to evaluate planck function.
    :type nu: float.
    :param beta: spectral index of the emissivity
    :type beta: float    
    :param T: temperature of black body in Kelvins.
    :type T: float.
    :return: float -- modified black body brightness.

    """    
    return (mbb(nu,beta,T)/mbb(nu0,beta,T))*psm.convert_units('MJysr','uK_CMB',nu)/psm.convert_units('MJysr','uK_CMB',nu0)

def PL_uK(nu,beta,nu0=23.):
    """Power law.

    :param nu: frequency in GHz at which to evaluate planck function.
    :type nu: float.
    :param beta: spectral index of the emissivity
    :type beta: float    
    :return: float -- power law brightness.

    """    
    return psm.convert_units('uK_RJ','uK_CMB',nu)/psm.convert_units('uK_RJ','uK_CMB',nu0)*(nu/nu0)**beta


def dmbbT(nu,T):
    x = const.h.value*nu*1.e9/const.k_B.value/T
    return (x/T)*np.exp(x)/np.expm1(x)

def dmbbp(nu,p):
    x = const.h.value*nu*1.e9/const.k_B.value
    return -x*np.exp(x*p)/np.expm1(x*p)

def ddmbbT(nu,T):
    x = const.h.value*nu*1.e9/const.k_B.value/T
    return (x*np.tanh(x/2)-2)*((x/T)*np.exp(x)/np.expm1(x))/T

def d3mbbT(nu,T):
    x = const.h.value*nu*1.e9/const.k_B.value/T
    theta = (x/T)*np.exp(x)/np.expm1(x)
    TR2= x*np.tanh(x/2)-2
    TR3= x**2*(np.cosh(x)+2)/(np.cosh(x)-1)
    return theta*(TR3+ 6*(1+TR2))/T/T
    
def Gaussian(x,mu,sigma):
    coeffnorm = 1/(sigma*np.sqrt(2*np.pi))
    coeffexp = ((x-mu)/sigma)**2
    return coeffnorm*np.exp(-coeffexp/2)

def from_ellnu_to_matrixnunu(DLdcflat,Nf,Nell):
    Ncross = int(Nf*(Nf+1)/2)
    DLdc = DLdcflat.reshape(Nell,Ncross)
    DLdcmatrix = np.zeros((Nf*Nell,Nf*Nell))
    a=0
    for i in range(Nf):
        for j in range(i,Nf):
            np.fill_diagonal(DLdcmatrix[i*Nell:(i+1)*Nell,j*Nell:(j+1)*Nell],DLdc[:,a])
            np.fill_diagonal(DLdcmatrix[j*Nell:(j+1)*Nell,i*Nell:(i+1)*Nell],DLdc[:,a])
            a = a+1
    return(DLdcmatrix)

def rad2deg(x):
    return x*180/np.pi

def deg2rad(x):
    return x*np.pi/180

################################
# Power spectrum from flat maps
################################

def _spectral_iso(data_sp, bins=None, sampling=1.0, return_counts=False):
    """
    Internal function.
    Check power_spectrum_iso for documentation details.
    Parameters
    ----------
    data_sp : TYPE
        DESCRIPTION.
    bins : TYPE, optional
        DESCRIPTION. The default is None.
    sampling : TYPE, optional
        DESCRIPTION. The default is 1.0.
    return_coutns : bool, optional
        Return counts per bin.
    Returns
    -------
    bins : TYPE
        DESCRIPTION.
    ps_mean : TYPE
        DESCRIPTION.
    ps_std : TYPE
        DESCRIPTION.
    counts : array, optional
        Return counts per bin return_counts=True.
    """
    N = data_sp.shape[0]
    ndim = data_sp.ndim
    # Build an array of isotropic wavenumbers making use of numpy broadcasting
    wn = (2 * np.pi * np.fft.fftfreq(N, d=sampling)).reshape((N,) + (1,) * (ndim - 1))
    wn_iso = np.zeros(data_sp.shape)
    for i in range(ndim):
        wn_iso += np.moveaxis(wn, 0, i) ** 2
    wn_iso = np.sqrt(wn_iso)
    # We do not need ND-arrays anymore
    wn_iso = wn_iso.ravel()
    data_sp = data_sp.ravel()
    # We compute associations between index and bins
    if bins is None:
        bins = np.sort(np.unique(wn_iso)) # Default binning
    index = np.digitize(wn_iso, bins) - 1
    # Stacking
    stacks = np.empty(len(bins), dtype=object)
    for i in range(len(bins)):
        stacks[i] = []
    for i in range(len(index)):
        if index[i] >= 0:
            stacks[index[i]].append(data_sp[i])
    counts = []
    # Computation for each bin of the mean power spectrum and standard deviations of the mean
    ps_mean = np.zeros(len(bins), dtype=data_sp.dtype) # Allow complex values (for cross-spectrum)
    ps_std = np.zeros(len(bins)) # If complex values, note that std first take the modulus
    for i in range(len(bins)):
        ps_mean[i] = np.mean(stacks[i])
        count = len(stacks[i])
        ps_std[i] = np.std(stacks[i]) / np.sqrt(count)
        counts.append(count)
    if return_counts:
        return bins, ps_mean, ps_std, np.array(counts)
    else:
        return bins, ps_mean, ps_std

def power_spectrum_iso(data, data2=None, bins=None, sampling=1.0, norm=None, return_counts=False):
    """
    Compute the isotropic power spectrum of input data.
    bins parameter should be a list of bin edges defining:
    bins[0] <= bin 0 values < bins[1]
    bins[1] <= bin 1 values < bins[2]
                ...
    bins[N-2] <= bin N-2 values < bins[N-1]
    bins[N-1] <= bin N-1 values
    Note that the last bin has no superior limit.
    Parameters
    ----------
    data : array
        Input data.
    bins : array, optional
        Array of bins. If None, we use a default binning which correspond to a full isotropic power spectrum.
        The default is None.
    sampling : float, optional
        Grid size. The default is 1.0.
    norm : TYPE, optional
        FFT normalization. Can be None or 'ortho'. The default is None.
    return_counts: bool, optional
        Return counts per bin. The default is None
    Raises
    ------
    Exception
        DESCRIPTION.
    Returns
    -------
    bins : TYPE
        DESCRIPTION.
    ps_mean : TYPE
        DESCRIPTION.
    ps_std : TYPE
        DESCRIPTION.
    counts : array, optional
        If return_counts=True, counts per bin.
    """
    # Check data shape
    for i in range(data.ndim):
        if data.shape[i] != data.shape[0]:
            raise Exception("Input data must be of shape (N, ..., N).")
    # Compute the full power spectrum of input data
    if data2 is None:
        data_ps = power_spectrum(data, norm=norm)
    else:
        data_ps = power_spectrum(data, data2=data2, norm=norm)
    return _spectral_iso(data_ps, bins=bins, sampling=sampling, return_counts=return_counts)

def power_spectrum(data, data2=None, norm=None):
    """
    Compute the full power spectrum of input data.
    Parameters
    ----------
    data : array
        Input data.
    norm : str
        FFT normalization. Can be None or 'ortho'. The default is None.
    Returns
    -------
    None.
    """
    if data2 is None:
        result=np.absolute(np.fft.fftn(data, norm=norm))**2
    else:
        result=np.real(np.conjugate(np.fft.fftn(data, norm=norm))*np.fft.fftn(data2, norm=norm))
    return result


def binning(ell,arr,bintab):

    ind = np.digitize(ell, bintab)
    arrbin = np.array([arr[ind == i].mean() for i in range(1, len(bintab))])
    # ellbin = np.array([ell[ind == i].mean() for i in range(1, len(bintab))])

    return arrbin


#Downgrade

def downgrade_alm(input_alm,nside_in,nside_out):
    """
    This is a Function to downgrade Alm correctly.
    nside_in must be bigger than nside_out.
    In this function, lmax_in = 3*nside_in-1 , lmax_out = 3*nside_out-1 .
    input_alm must be lmax = lmax_in and output_alm must be lmax = lmax_out.
    This function get only values in the range 0 < l < lmax_out from input_alm,
    and put these values into output_alm which has range 0 < l < lmax_out.
    """
    lmax_in = nside_in*3-1
    lmax_out = nside_out*3-1
    output_alm = np.zeros((3,hp.sphtfunc.Alm.getsize(lmax_out)),dtype=complex)
    
    for m in range(lmax_out+1):
        idx_1_in = hp.sphtfunc.Alm.getidx(lmax_in,m ,m)
        idx_2_in = hp.sphtfunc.Alm.getidx(lmax_in,lmax_out ,m)

        idx_1_out = hp.sphtfunc.Alm.getidx(lmax_out,m ,m)
        idx_2_out = hp.sphtfunc.Alm.getidx(lmax_out,lmax_out ,m)

        output_alm[:,idx_1_out:idx_2_out+1] = input_alm[:,idx_1_in:idx_2_in+1]
    return output_alm

def downgrade_map(input_map,nside_out,nside_in=512):
    """
    This is a Function to downgrade map correctly in harmonic space.
    nside_in must be bigger than nside_out.
    input_map must have nside_in.
    output_map has nside_out as Nside
    """
    #  nside_in= hp.npix2nside(len(input_map))
    input_alm = hp.map2alm(input_map)  #input map → input alm
    output_alm = downgrade_alm(input_alm,nside_in,nside_out) # input alm → output alm (decrease nside)
    output_map = hp.alm2map(output_alm,nside=nside_out)#  output alm → output map
    return output_map

def MBBpysm(freq,A,beta,T,nu0):
    #A in muKCMB
    factor= u.K_RJ.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq*u.GHz))/u.K_RJ.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(nu0*u.GHz))
    mapd=np.array([A*mbb(freq[f],beta-2,T)/mbb(nu0,beta-2,T)*factor[f] for f in range(len(freq))])
    return mapd

def MBB_fit(freq,beta,T):
    #A in muKCMB
    nu0=353
    factor= u.K_RJ.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq*u.GHz))
    mapd=mbb(freq,beta-2,T)*factor
    return mapd
    
def dBnu_dT(nu,T):
    return (B(nu,T)*constc/nu/T)**2 / 2 * np.exp(consth*nu*1e9/constk/T) / constk


#compute automatically moment's SEDs

import sympy as sym
import scipy.constants as constants

def model_mbb_moments(nside,nu,model,mom,tempmap,nu0=353.,maxborder=3,maxtorder=3,nside_moments=512,mult_factor=1.):
    npix_moments = hp.nside2npix(nside_moments)
    map3D = np.zeros([3,npix_moments])
    
    beta = sym.Symbol('ß')
    T = sym.Symbol('T')
    
    nuval = nu * 1e9
    nu0val = nu0 * 1e9
    Bval = 2*const.h.value*(nuval**3)/const.c.value**2
    Cval = const.h.value*nuval/const.k_B.value
    Bval0 = 2*const.h.value*(nu0val**3)/const.c.value**2
    Cval0 = const.h.value*nu0val/const.k_B.value
    Bvalratio = Bval/Bval0
    mbb = ((nuval / nu0val) ** beta) * Bvalratio / (sym.exp(Cval / T) - 1) * (sym.exp(Cval0 / T) - 1)
    for border in range(maxborder+1):
        for torder in range(maxtorder+1-border):
            print((torder,border))
            analyticalmom = sym.diff(mbb,beta,border)*sym.diff(mbb,T,torder).factor()/mbb**2
        
            if torder == 0:
                valuemom = float(analyticalmom)
            else:
                analyticalmom = sym.lambdify(T,analyticalmom,'numpy')
                valuemom = analyticalmom(tempmap)

            if ((border == 0)  * (torder == 0)) == 1:
                modelcomplex = (model[1]+1j*model[2]) * 1./(np.math.factorial(border)*np.math.factorial(torder))*mom[1,border,torder]*valuemom
                map3D[0] += model[0] * 1./(np.math.factorial(border)*np.math.factorial(torder))*np.real(mom[0,border,torder])*valuemom
            else:
                modelcomplex = (model[1]+1j*model[2]) * mult_factor/(np.math.factorial(border)*np.math.factorial(torder))*mom[1,border,torder]*valuemom
                map3D[0] += model[0] * mult_factor/(np.math.factorial(border)*np.math.factorial(torder))*np.real(mom[0,border,torder])*valuemom
            map3D[1] += np.real(modelcomplex)
            map3D[2] += np.imag(modelcomplex)
    if nside != nside_moments: map3D = hp.ud_grade(map3D,nside)
    return map3D

def symbolic_derivative_mbb(order, var):
    """
    Calcule les dérivées analytiques du corps noir modifié (MBB).
    Parameters:
        order : int
            Ordre de la dérivée (1, 2, ...).
        var : str
            Variable par rapport à laquelle on dérive ('T', '1/T', ou 'beta').
    Returns:
        sympy expression : Dérivée symbolique normalisée du MBB.
    """
    # Définir les variables symboliques
    nu, T, beta, nu0 = sp.symbols('nu T beta nu0', real=True, positive=True)
    h, c, k = sp.symbols('h c k', real=True, positive=True)

    # Corps noir modifié
    x = h * nu / (k * T)  # Argument sans dimension
    x0 = h * nu0 / (k * T)  # Argument sans dimension
    Bnu = (2 * h * nu**3 / c**2) / (sp.exp(x) - 1)
    Bnu0 = (2 * h * nu0**3 / c**2) / (sp.exp(x0) - 1)

    # Intensité normalisée
    if var in ['T', '1/T']:
        I_nu = Bnu / Bnu0
    elif var == 'beta':
        I_nu = (nu / nu0)**beta
    else:
        raise ValueError("La variable doit être 'T', '1/T', ou 'beta'.")

    # Calcul des dérivées selon la variable spécifiée
    if var == 'T':
        variable = T
        derivative = sp.diff(I_nu, variable, order)
    elif var == '1/T':
        y = sp.symbols('y', real=True, positive=True)  # Définir y = 1/T
        I_nu_y = I_nu.subs(T, 1 / y)  # Substituer T par 1/y
        derivative = sp.diff(I_nu_y, y, order).subs(y, 1 / T)  # Revenir à T
    elif var == 'beta':
        variable = beta
        derivative = sp.diff(I_nu, variable, order)

    # Normalisation par l'intensité I_nu
    return sp.simplify(derivative / I_nu)
    