from astropy import constants as const
import numpy as np
import healpy as hp
import pysm3.units as u
import sympy as sym
import scipy.constants as constants
from astropy.cosmology import Planck18 as cosmo

# Convert units

def uK_RJ_to_MJy_sr(nu):
    """
    Compute factors to convert brightness from uK_RJ to MJy/sr for all input frequencies.

    Parameters
    ----------
    nu : float or np.array
        Frequencies for which conversion factors should be computed in GHz.

    Returns
    -------
    float or np.array
        Conversion factors for all input frequencies.

    """
    k = const.k_B.value
    c = const.c.value
    
    return 2*k * (nu*1e9 / c)**2 * 1e11

def uK_RJ_to_uK_CMB(nu):
    """
    Compute factors to convert brightness from uK_RJ to uK_CMB for all input frequencies.

    Parameters
    ----------
    nu : float or np.array
        Frequencies for which conversion factors should be computed in GHz.

    Returns
    -------
    float or np.array
        Conversion factors for all input frequencies.

    """
    h = const.h.value
    k = const.k_B.value
    Tcmb = cosmo.Tcmb0.value
    
    x = h*nu*1e9 / (k*Tcmb)
    return (np.exp(x) - 1)**2 / (x**2 * np.exp(x))

def unit_conversion(nu, input_unit, output_unit):
    """
    Compute factors to convert brightness from input_unit to output_unit for all input frequencies.

    Parameters
    ----------
    nu : float or np.array
        Frequencies for which conversion factors should be computed in GHz.
    input_unit : string
        Unit from which conversion factors should be computed. Can be 'MJy/sr', 'uK_RJ' or 'uK_CMB'.
    output_unit : string
        Unit to which conversion factors should be computed. Can be 'MJy/sr', 'uK_RJ' or 'uK_CMB'.

    Returns
    -------
    float or np.array
        Conversion factors for all input frequencies.

    """
    if input_unit == 'uK_CMB':
        if output_unit == 'uK_CMB':
            return np.ones_like(nu)
        
        elif output_unit == 'uK_RJ':
            return 1 / uK_RJ_to_uK_CMB(nu)
        
        elif output_unit == 'MJy/sr':
            return uK_RJ_to_MJy_sr(nu) / uK_RJ_to_uK_CMB(nu)
        
        else:
            raise ValueError('Incorrect output unit')
            
    elif input_unit == 'uK_RJ':
        if output_unit == 'uK_CMB':
            return uK_RJ_to_uK_CMB(nu)
        
        elif output_unit == 'uK_RJ':
            return np.ones_like(nu)
        
        elif output_unit == 'MJy/sr':
            return uK_RJ_to_MJy_sr(nu)
        
        else:
            raise ValueError('Incorrect output unit')
            
    elif input_unit == 'MJy/sr':
        if output_unit == 'uK_CMB':
            return uK_RJ_to_uK_CMB(nu) / uK_RJ_to_MJy_sr(nu)
        
        elif output_unit == 'uK_RJ':
            return 1 / uK_RJ_to_MJy_sr(nu)
        
        elif output_unit == 'MJy/sr':
            return np.ones_like(nu)
        
        else:
            raise ValueError('Incorrect output unit')
    
    else:
        raise ValueError('Incorrect input unit')

def bandpass_unit_conversion(nu, input_unit, output_unit):
    """
    Compute factors to convert brightness from input_unit to output_unit for all input frequencies assuming top-hat bandpasses if needed.

    Parameters
    ----------
    nu : float or np.array
        Frequency bands for which conversion factors should be computed in GHz.
    input_unit : string
        Unit from which conversion factors should be computed. Can be 'MJy/sr', 'uK_RJ' or 'uK_CMB'.
    output_unit : string
        Unit to which conversion factors should be computed. Can be 'MJy/sr', 'uK_RJ' or 'uK_CMB'.

    Returns
    -------
    float or np.array
        Conversion factors for all input frequency bands.
    """
    if nu.ndim == 1:
        factors = unit_conversion(nu, input_unit, output_unit)
    
    else:
        Ngrid = nu.shape[1]
        weights = np.ones_like(nu)
        bw = np.max(nu, axis=1) - np.min(nu, axis=1)
        weights /= np.tile(bw, [Ngrid,1]).T
        weights_to_in = unit_conversion(nu, 'MJy/sr', input_unit)
        weights_to_out = unit_conversion(nu, 'MJy/sr', output_unit)
        factors = np.trapezoid(weights_to_out, nu) / np.trapezoid(weights_to_in, nu)

    return factors

#FONCTIONS

def B(nu,b_T):
    """
    Planck function.

    Parameters
    ----------

    :param nu: frequency in GHz at which to evaluate planck function.
    :type nu: float.
    :param b_T: inverse temperature (coldness, 1/T) of black body in Kelvins^(-1).
    :type b_T: float.
    :return: float -- black body brightness.

    """
    x = const.h.value*nu*1.e9*b_T/const.k_B.value
    return 2.*const.h.value *(nu *1.e9)**3/ const.c.value**2/np.expm1(x)

def mbb(nu,beta,b_T):
    """Modified black body.

    :param nu: frequency in GHz at which to evaluate planck function.
    :type nu: float.
    :param beta: spectral index of the emissivity
    :type beta: float    
    :param b_T: inverse temperature (coldness, 1/T) of black body in Kelvins^(-1).
    :type b_T: float.
    :return: float -- modified black body brightness.

    """    
    return B(nu,b_T)*(1e9*nu)**beta


def mbb_uK(nu,beta,b_T,nu0=353.):
    """Modified black body.

    :param nu: frequency in GHz at which to evaluate planck function.
    :type nu: float.
    :param beta: spectral index of the emissivity
    :type beta: float    
    :param b_T: inverse temperature (coldness, 1/T) of black body in Kelvins^(-1).
    :type b_T: float..
    :return: float -- modified black body brightness.

    """    
    S_nu = (mbb(nu, beta, b_T) / mbb(nu0, beta, b_T))
    
    if nu.ndim == 2:
        Ngrid = nu.shape[1]
        weights = np.ones_like(nu)
        bw = np.max(nu, axis=1) - np.min(nu, axis=1)
        weights /= np.tile(bw, [Ngrid,1]).T
        S_nu = np.trapezoid(S_nu * weights, nu)

    return S_nu * bandpass_unit_conversion(nu, 'MJy/sr', 'uK_CMB') / bandpass_unit_conversion(nu0, 'MJy/sr', 'uK_CMB')

def PL_uK(nu,beta,nu0=23.):
    """Power law.

    :param nu: frequency in GHz at which to evaluate planck function.
    :type nu: float.
    :param beta: spectral index of the emissivity
    :type beta: float    
    :return: float -- power law brightness.

    """    
    S_nu = (nu/nu0)**beta

    if nu.ndim == 1:
        return S_nu * bandpass_unit_conversion(nu, 'uK_RJ', 'uK_CMB') / bandpass_unit_conversion(nu0, 'uK_RJ', 'uK_CMB')
    
    else:
        S_nu *= unit_conversion(nu, 'uK_RJ', 'MJy/sr') / unit_conversion(nu0, 'uK_RJ', 'MJy/sr')
        Ngrid = nu.shape[1]
        weights = np.ones_like(nu)
        bw = np.max(nu, axis=1) - np.min(nu, axis=1)
        weights /= np.tile(bw, [Ngrid,1]).T
        S_nu = np.trapezoid(S_nu * weights, nu)

        return S_nu * bandpass_unit_conversion(nu, 'MJy/sr', 'uK_CMB') / bandpass_unit_conversion(nu0, 'MJy/sr', 'uK_CMB')

def dmbbT(nu,T):
    '''first order derivative of black body with respect to T '''
    x = const.h.value*nu*1.e9/const.k_B.value/T
    dS_nu = (x/T)*np.exp(x)/np.expm1(x)

    if nu.ndim == 2:
        Ngrid = nu.shape[1]
        weights = np.ones_like(nu)
        bw = np.max(nu, axis=1) - np.min(nu, axis=1)
        weights /= np.tile(bw, [Ngrid,1]).T
        dS_nu = np.trapezoid(dS_nu * weights, nu)

    return dS_nu

def dmbb_bT(nu,p):
    '''first order derivative of black body with respect to 1/T '''
    x = const.h.value*nu*1.e9/const.k_B.value
    dS_nu = -x*np.exp(x*p)/np.expm1(x*p)

    if nu.ndim == 2:
        Ngrid = nu.shape[1]
        weights = np.ones_like(nu)
        bw = np.max(nu, axis=1) - np.min(nu, axis=1)
        weights /= np.tile(bw, [Ngrid,1]).T
        dS_nu = np.trapezoid(dS_nu * weights, nu)

    return dS_nu

def ddmbbT(nu,T):
    '''second order derivative of black body with respect to T '''
    x = const.h.value*nu*1.e9/const.k_B.value/T
    d2S_nu = (x*np.tanh(x/2)-2)*((x/T)*np.exp(x)/np.expm1(x))/T

    if nu.ndim == 2:
        Ngrid = nu.shape[1]
        weights = np.ones_like(nu)
        bw = np.max(nu, axis=1) - np.min(nu, axis=1)
        weights /= np.tile(bw, [Ngrid,1]).T
        d2S_nu = np.trapezoid(d2S_nu * weights, nu)

    return d2S_nu

def d3mbbT(nu,T):
    '''third order derivative of black body with respect to T '''
    x = const.h.value*nu*1.e9/const.k_B.value/T
    theta = (x/T)*np.exp(x)/np.expm1(x)
    TR2= x*np.tanh(x/2)-2
    TR3= x**2*(np.cosh(x)+2)/(np.cosh(x)-1)
    d3S_nu = theta*(TR3+ 6*(1+TR2))/T/T

    if nu.ndim == 2:
        Ngrid = nu.shape[1]
        weights = np.ones_like(nu)
        bw = np.max(nu, axis=1) - np.min(nu, axis=1)
        weights /= np.tile(bw, [Ngrid,1]).T
        d3S_nu = np.trapezoid(d3S_nu * weights, nu)

    return d3S_nu
    
def Gaussian(x,mu,sigma):
    '''gaussian curve '''
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
    Compute the analytical derivatives of modified black body function 
    Parameters:
        order : int
            Ordre de la dérivée (1, 2, ...).
        var : str
            Variable par rapport à laquelle on dérive ('T', '1/T', ou 'beta').
    Returns:
        sympy expression : Dérivée symbolique normalisée du MBB.
    """
    nu, T, beta, nu0 = sym.symbols('nu T beta nu0', real=True, positive=True)
    h, c, k = sym.symbols('h c k', real=True, positive=True)
    x = h * nu / (k * T)  
    x0 = h * nu0 / (k * T)  
    Bnu = (2 * h * nu**3 / c**2) / (sym.exp(x) - 1)
    Bnu0 = (2 * h * nu0**3 / c**2) / (sym.exp(x0) - 1)

    if var in ['T', '1/T']:
        I_nu = Bnu / Bnu0
    elif var == 'beta':
        I_nu = (nu / nu0)**beta
    else:
        raise ValueError("The variable must be 'T', '1/T', or 'beta'.")

    if var == 'T':
        variable = T
        derivative = sym.diff(I_nu, variable, order)
    elif var == '1/T':
        y = sym.symbols('y', real=True, positive=True)  
        I_nu_y = I_nu.subs(T, 1 / y)  
        derivative = sym.diff(I_nu_y, y, order).subs(y, 1 / T)  
    elif var == 'beta':
        variable = beta
        derivative = sym.diff(I_nu, variable, order)
    return sym.simplify(derivative / I_nu)
    
