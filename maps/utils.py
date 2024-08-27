import numpy as np, sympy as sp, scipy.special as scsp, pandas as pd

import pickle, healpy as hp

import numpy.random as nr, scipy.stats as scst
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from enterprise.signals import anis_coefficients as ac

import sympy

import scipy.linalg as sl

from . import clebschGordan as CG
from . import anis_pta as ap

from scipy.interpolate import interp1d
from astroML.linear_model import LinearRegression


def invert_omega(hp_map):
    """A function to change between GW propogation direction and GW source direction.

    This function takes a healpy map or maps which has either GW propogation direction 
    or GW source direction and swaps to the other.  This function supports lists or 
    arrays of healpy maps indicated to the function with len(hp_map.shape)>1.

    Args:
        hp_map (np.ndarray or list): A healpy map or array or list of healpy maps

    Returns:
        list: A list of healpy maps or a single healpy map with inverted direction
    """
    arg_arr = np.array(hp_map)
    if len(arg_arr.shape) == 1:
        arg_arr = arg_arr[None, :]

    inv_map = []
    for mp in arg_arr:
        nside = hp.get_nside(mp)

        all_pix_idx = np.array([ ii for ii in range(hp.nside2npix(nside = nside)) ])

        all_pix_x, all_pix_y, all_pix_z = hp.pix2vec(nside = nside, ipix = all_pix_idx)

        inv_all_pix_x = -1 * all_pix_x
        inv_all_pix_y = -1 * all_pix_y
        inv_all_pix_z = -1 * all_pix_z

        inv_pix_idx = [hp.vec2pix(nside = nside, x = xx, y = yy, z = zz) for (xx, yy, zz) in zip(inv_all_pix_x, inv_all_pix_y, inv_all_pix_z)]
        inv_map.append(mp[inv_pix_idx])

    if len(inv_map) == 1:
        return inv_map[0]
    
    return inv_map

    
def convert_blm_params_to_clm(pta_anis, blm_params):
    """A function to convert a set of blm parameters to clm parameters.

    Args:
        pta_anis (anis_pta.pta_anis): The anisotropic PTA object
        blm_params (np.ndarray): The array of b_lm parameters

    Returns:
        clms: The array of c_lm parameters
    """

    blm = pta_anis.sqrt_basis_helper.blm_params_2_blms(blm_params[1:])

    #Note that when using this mode, the supplied params need to be in
    #the healpy format

    new_clms = pta_anis.sqrt_basis_helper.blm_2_alm(blm)

    #Only need m >= 0 entries for clmfromalm
    clms_rvylm = pta_anis.clmFromAlm(new_clms)
    clms_rvylm = clms_rvylm * np.sqrt(4 * np.pi) / clms_rvylm[0] #Normalize such that c_00 to sqrt(4pi)
    #clms_rvylm[0] *= 12.56637061

    return clms_rvylm


def signal_to_noise(pta, lm_params = None):
    """A function to compute the SNR of the square-root spherical harmonic anisotropy.

    This function computes the signal-to-noise ratio of anisotropy in the square-root 
    spherical harmonic model. This function computes equation 17 from the paper
    Pol, Taylor, Romano 2022. 
    NOTE: This function only works with the square-root spherical harmonic model.

    Args:
        pta (anis_pta.anis_pta): The anis_pta object with the signal model
        lm_params (lmfit.Minimizer.minimize, optional): The lmfit parameters of the
            minimized solution. Setting to None will compute this inside the function.
            Defaults to None.

    Returns:
        tuple: A tuple containing:
            total_sn (float): The total signal-to-noise ratio
            iso_sn (float): The isotropic signal-to-noise ratio
            anis_sn (float): The anisotropic signal-to-noise ratio
    """

    if lm_params is None:
        lm_out = pta.max_lkl_sqrt_power()
    else:
        lm_out = lm_params

    pta_mono = ap.anis_pta(pta.psrs_theta, pta.psrs_phi, pta.xi, pta.rho, pta.sig, 
                 os = 1, pair_cov = pta.pair_cov, l_max = 0, nside = pta.nside, 
                 mode = pta.mode, use_physical_prior = pta.use_physical_prior, 
                 include_pta_monopole = pta.include_pta_monopole, 
                 pair_idx = pta.pair_idx) #OS already applied in pta

    lm_out_mono = pta_mono.max_lkl_sqrt_power()

    lp = np.array(list(lm_out.params.valuesdict().values()))
    lp_mono = np.array(list(lm_out_mono.params.valuesdict().values()))

    if pta.include_pta_monopole:
        opt_clm = convert_blm_params_to_clm(pta, lp[2:])
        opt_clm_mono = convert_blm_params_to_clm(pta_mono, lp_mono[2:])
    else:
        opt_clm = convert_blm_params_to_clm(pta, lp[1:])
        opt_clm_mono = convert_blm_params_to_clm(pta_mono, lp_mono[1:])

    if pta.include_pta_monopole:
        ml_orf = pta.orf_from_clm(np.append((lp[1]), opt_clm)) + (10 ** lp[0]) * 0.5
        hd_orf = pta_mono.orf_from_clm(np.append((lp_mono[1]), opt_clm_mono)) + (10 ** lp_mono[0]) * 0.5
    else:
        ml_orf = pta.orf_from_clm(np.append((lp[0]), opt_clm))
        hd_orf = pta_mono.orf_from_clm(np.append((lp_mono[0]), opt_clm_mono))

    snm = np.sum(-1 * (pta.rho - ml_orf) ** 2 / (2 * (pta.sig) ** 2))
    nm = np.sum(-1 * (pta.rho) ** 2 / (2 * (pta.sig) ** 2))
    hdnm = np.sum(-1 * (pta.rho - hd_orf) ** 2 / (2 * (pta.sig) ** 2))

    total_sn = 2 * (snm - nm)
    iso_sn = 2 * (hdnm - nm)
    anis_sn = 2 * (snm - hdnm)

    return total_sn, iso_sn, anis_sn


def angular_power_spectrum(clm):
    """A function to compute the angular power spectrum from the spherical harmonic coefficients.

    This function computes the angular power spectrum from the spherical harmonic coefficients
    by taking the average of the square of the coefficients for each l.

    Args:
        clm (np.ndarray): The array of spherical harmonic coefficients

    Returns:
        C_l: The angular power spectrum per spherical harmonic l.
    """

    maxl = int(np.sqrt(clm.shape[0]))
    new_clm2 = clm ** 2

    C_l = np.zeros((maxl))
    idx = 0

    for ll in range(maxl):
        if ll == 0:
            C_l[ll] = new_clm2[ll]
            idx += 1
        else:
            subset_len = 2 * ll + 1
            subset = np.arange(idx, idx + subset_len)

            C_l[ll] = np.sum(new_clm2[subset]) / (2 * ll + 1)
            idx = subset[-1]

    return C_l


def draw_random_sample(ip_arr, bins = 50, nsamp = 10):
    """A function to draw a random sample from a distribution using inverse transform sampling.

    This function draws nsamp random samples from a user supplied distribution by
    first computing the cdf, then transforming uniform random numbers into draws 
    from the original distribution.

    Args:
        ip_arr (np.ndarray): The input distribution to sample from.
        bins (int): The number of value bins to use. Use larger values for larger distributions. 
            Defaults to 50.
        nsamp (int): The number of random samples to draw. Defaults to 10.

    Returns:
        np.ndarray: The random samples drawn from the input distribution.
    """

    counts, bin_ed = np.histogram(ip_arr, bins = bins, density = True)
    bin_mid = (bin_ed[1:] + bin_ed[:-1]) / 2.

    cdf = np.cumsum(counts) / np.sum(counts)

    interp_func = interp1d(cdf, bin_mid)

    rn_draw = nr.uniform(low = cdf.min(), high = cdf.max(), size = nsamp)

    return interp_func(rn_draw)


def posterior_sampled_Cl_skymap(anis_pta, chain, burn = 0, n_draws = 100):
    """A function to generate posterior sampled sky maps from a chain file.

    Return collection of sky maps randomly drawn from posterior
    given an (emcee) chain file sampling the posterior around the
    maximum likelihood value.

    Args:
        anis_pta (anis_pta.anis_pta): The anisotropic PTA object
        chain (pd.DataFrame): chain file with posterior samples with rows = samples, params = columns
        burn (int, optional): burn-in length. Defaults to 0.
        n_draws (int): number of draws from chain file to generate the posterior averaged sky map.
            Defaults to 100.

    Returns:
        tuple: A tuple containing:
            pow_map (np.ndarray): (n_draws x n_pixel) numpy array of maps corresponding 
                to n_draws from posterior.
            Cl (np.ndarray): (n_draws x n_Cl) numpy array of C_l values corresponding 
                to n_draws from posterior.
    """

    chain_copy = chain.copy()

    if anis_pta.include_pta_monopole:
        chain_copy.insert(loc = 2, column = 'b_00', value = 1)
    else:
        chain_copy.insert(loc = 1, column = 'b_00', value = 1)

    sub_chain = chain_copy.sample(n = n_draws)

    pow_maps = np.full((n_draws, hp.pixelfunc.nside2npix(anis_pta.nside)), 0.0)
    post_Cl = np.full((n_draws, anis_pta.l_max + 1), 0.0)

    for ii in range(n_draws):

        if anis_pta.include_pta_monopole:
            clms = convert_blm_params_to_clm(anis_pta, sub_chain.iloc[ii, 2:])
        else:
            clms = convert_blm_params_to_clm(anis_pta, sub_chain.iloc[ii, 1:])

        Cl = angular_power_spectrum(anis_pta, clms)

        if anis_pta.include_pta_monopole:
            pow_maps[ii] = (10 ** sub_chain.iloc[ii, 1]) * ac.mapFromClm(clms, nside = anis_pta.nside)
        else:
            pow_maps[ii] = (10 ** sub_chain.iloc[ii, 0]) * ac.mapFromClm(clms, nside = anis_pta.nside)

        post_Cl[ii] = Cl

    return pow_maps, post_Cl


def woodbury_inverse(A, U, C, V, ret_cond = False):
    """A function to compute the inverse of a matrix using the Woodbury matrix identity.

    This function computes the inverse of a matrix using the Woodbury matrix identity 
    for more stable matrix inverses. The matrix labels are the same as those used 
    in its wikipedia page (https://en.wikipedia.org/wiki/Woodbury_matrix_identity)
    This solves the inverse to: (A + UCV)^-1.
    
    This function can also solve inverses of the form: (A + K)^-1 where A is a
    diagonal matrix and K is a dense matrix. To do this, simply set U and C to 
    the identity matrix and V to K.
    
    Args:
        A (np.ndarray): An invertible nxn matrix
        U (np.ndarray): A nxk matrix
        C (np.ndarray): An invertible kxk matrix
        V (np.ndarray): A kxn matrix
        ret_cond (bool, optional): Whether to return the condition number of the matrix (C + V A^-1 U)

    Returns:
        np.ndarray: The inverse of the matrix (A + UCV)^-1
    """

    Ainv = np.diag( 1/np.diag(A) )
    Cinv = np.linalg.pinv(C)

    # (A+UCV)^-1 = (A^-1) - A^-1 @ U @ (C^-1 + V @ A^-1 @ U)^-1 @ V @ A^-1
    CVAU = Cinv + V @ Ainv @ U
    tot_inv = Ainv - Ainv @ U @ np.linalg.solve(CVAU, V @ Ainv) 

    if ret_cond:
        return tot_inv, np.linalg.cond(CVAU)
    
    return tot_inv
