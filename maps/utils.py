import numpy as np, sympy as sp, scipy.special as scsp, pandas as pd

import pickle, healpy as hp

import numpy.random as nr, scipy.stats as scst
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from enterprise.signals import anis_coefficients as ac

import sympy

import scipy.linalg as sl

#from . import clebschGordan as CG
#from . import anis_pta as ap
import clebschGordan as CG
import anis_pta as ap

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


def signal_to_noise(pta, lm_params = None, pair_cov = False, method = 'leastsq'):
    """A function to compute the SNR of the square-root spherical harmonic anisotropy.

    This function computes the signal-to-noise ratio of anisotropy in the square-root 
    spherical harmonic model. This function computes equation 17 from the paper
    Pol, Taylor, Romano 2022. 
    NOTE: If using pair covariance, the noise model will ignore this, as the null 
    hypothesis has uncorrelated noise.
    NOTE: This function only works with the square-root spherical harmonic model.
    NOTE: This function returns the square of the signal-to-noise ratio!

    Args:
        pta (anis_pta.anis_pta): The anis_pta object with the signal model
        lm_params (lmfit.Minimizer.minimize, optional): The lmfit parameters of the
            minimized solution. Setting to None will compute this inside the function.
            Defaults to None.
        use_pair_cov (bool): Whether to include the pair covariance matrix. 
            Defaults to False.

    Returns:
        tuple: A tuple containing:
            total_sn (float): The squared total signal-to-noise ratio
            iso_sn (float): The squared isotropic signal-to-noise ratio
            anis_sn (float): The squared anisotropic signal-to-noise ratio
    """

    if lm_params is None:
        lm_out = pta.max_lkl_sqrt_power(pair_cov=pair_cov,method=method)
    else:
        lm_out = lm_params

    iso_pta = ap.anis_pta(pta.psrs_theta, pta.psrs_phi, pta.xi, pta.rho, pta.sig, 
                 os = 1, pair_cov = pta.pair_cov, l_max = 0, nside = pta.nside, 
                 mode = pta.mode, use_physical_prior = pta.use_physical_prior, 
                 include_pta_monopole = pta.include_pta_monopole, 
                 pair_idx = pta.pair_idx) #OS already applied in pta

    lm_out_iso = iso_pta.max_lkl_sqrt_power(pair_cov=pair_cov,method=method)

    mini = np.array(list(lm_out.params.valuesdict().values()))
    iso_mini = np.array(list(lm_out_iso.params.valuesdict().values()))

    # Convert blm to clm
    if pta.include_pta_monopole:
        A_mono = 10**mini[0]
        A2 = 10**mini[1]
        iso_A_mono = 10**iso_mini[0]
        iso_A2 = 10**iso_mini[1]

        clm = convert_blm_params_to_clm(pta, mini[2:])
        iso_clm = convert_blm_params_to_clm(iso_pta, iso_mini[2:])

    else:
        A_mono = 0
        A2 = 10**mini[0]
        iso_A_mono = 0
        iso_A2 = 10**iso_mini[0]

        clm = convert_blm_params_to_clm(pta, mini[1:])
        iso_clm = convert_blm_params_to_clm(iso_pta, iso_mini[1:])

    ani_orf = A_mono + A2*pta.orf_from_clm(clm, include_scale=False)
    iso_orf = iso_A_mono + iso_A2*iso_pta.orf_from_clm(iso_clm, include_scale=False) 


    if pair_cov:
        if pta.pair_cov is None:
            raise ValueError("Pair covariance matrix is not set.")
        covinv = pta.pair_cov_N_inv
        noiseinv = pta.pair_ind_N_inv
    else:
        covinv = pta.pair_ind_N_inv
        noiseinv = pta.pair_ind_N_inv
    
    ani_res = pta.rho - ani_orf
    iso_res = pta.rho - iso_orf

    snm = (-1/2)*((ani_res).T @ covinv @ (ani_res)) # Anisotropy chi-square
    hdnm = (-1/2)*((iso_res).T @ covinv @ (iso_res)) # Isotropy chi-square
    nm = (-1/2)*((pta.rho).T @ noiseinv @ (pta.rho)) # Null chi-square (Not pair covariant)

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


# A handy function to do some anisotropy injection stuff
def inject_anisotropy(anis_pta, method='power_basis', sim_clm_wo_00=None, sim_log10_A2=0.0, sim_sig=0.01, pair_cov=False, seed=42, 
                      sim_power=50, sim_lon=270, sim_lat=45, sim_pixel_radius=10, return_vals=False):

    """ A handy function to create a sky with injected anisotropy. 
    This function upon completion creates 'injected' instances of the anis_pta object.

    Args:
        anis_pta (object) : The anis_pta instance created by MAPS.
        method (str, optional): The way of creating the inject sky. 'power_basis' or 'pixel'. 
            Defaults to 'power_basis'.
        sim_clm_wo_00 (list or np.ndarray): The list or array of clm values to inject for 'clm' method. 
            Should be a list/array of (clm_size - 1) length. 
            NOTE: c_00 is fixed internally to root(4pi).
        sim_log10_A2 (float): The amplitude correction to assume. Defaults to 0.0
        sim_sig (float): The cross-correlation uncertainty to assume. 
            Defaults to 0.01.
        pair_cov (bool): Whether to return the pair covariance matrix. 
            Defaults to False.
        seed (int): The seed to be passed to numpy random number generator for 'clm' method. 
            Defaults to 42.

        sim_power (int ot float): The power of anisotropy to inject in the sky for 'pixel' method. 
            Defaults to 50.
        sim_lon, sim_lat (int or float): The location of injection for 'pixel' method. 
            Defaults to 270, 45 respectively.
        sim_pixel_radius (int or float): The size of the pixel injection for 'pixel' method. 
            Defaults to 10.
        return_vals (bool, optional): Whether to return a tuple of injected parameters/power, rho and sig. 
            Defaults to None.

    Returns:
        tuple (optional): A tuple of 4 np.ndarrays containing the injected log10_A2 and clm values 
            if method is 'power_basis' or the injected power map if method is 'pixel', 
            injected cross-correlations, injected cross-correlation uncertainties and 
            injected pair covariance matrix (None if pair_cov=False).
            NOTE: The function also creates 'injected' instances of the anis_pta.

    Raises:
        ValueError: If clm values (without c_00) not specified when method='power_basis'.
        ValueError: If the size of sim_clm_wo_00 does not match the clm_size without c_00.
        ValueError: If method not in ['power_basis', 'pixel'].

    """

    if method == 'power_basis':

        # If sim_clm not specified
        if sim_clm_wo_00 is None:
            raise ValueError("Specify clm values (without c_00) in sim_clm_wo_00 to do a power_basis injection!")
        # Check if the specified clm's include c_00 or more clm_values
        if len(sim_clm_wo_00) == anis_pta.clm_size-1:
            # Without c_00
            clm_construct = {f"c_{i}{j}": 0.0 for i in range(1, anis_pta.l_max+1) for j in range(-i, i+1)}
            inj_dict = {"log10_A2": sim_log10_A2, "c_00" : np.sqrt(4*np.pi), 
                        **{key : sim_clm_wo_00[k] for k, key in enumerate(clm_construct)}}
        else:
            raise ValueError("The specified clm values either include c_00 or does not match the clm_size specified by l_max in anis_pta.")
        # A numpy array to work with
        inj_arr = np.array([inj_dict[i] for i in inj_dict])
        A2_inject = 10 ** inj_arr[0]

        inj_orf = A2_inject * (anis_pta.Gamma_lm.T @ inj_arr[1:, np.newaxis])
        # Simulate sig
        if pair_cov:
            inj_pair_cov = np.diag(np.repeat(sim_sig, repeats=anis_pta.npairs))
        else:
            inj_pair_cov = None
        inj_sig = np.repeat(sim_sig, repeats=anis_pta.npairs)

        # Simulate rho - Shift by mean and scale by std
        rng = nr.default_rng(seed=seed)
        normal_dist = rng.normal(size=anis_pta.npairs)
        inj_rho = inj_orf.reshape(anis_pta.npairs) + inj_sig*normal_dist

        anis_pta.injected_params = inj_dict
        anis_pta.injected_rho = inj_rho
        anis_pta.injected_sig = inj_sig
        anis_pta.injected_pair_cov = inj_pair_cov

        if return_vals:
            return inj_dict, inj_rho, inj_sig, inj_pair_cov


    elif method == 'pixel':

        # From MAPS example
        # Simulate an isotropic background plus a hotspot
        input_map = np.ones(anis_pta.npix)
        vec = hp.ang2vec(sim_lon, sim_lat, lonlat=True)
        radius = np.radians(sim_pixel_radius)

        disk_anis = hp.query_disc(nside = anis_pta.nside, vec = vec, radius = radius, inclusive = False)
    
        input_map[disk_anis] += sim_power

        # Simulate rho
        inj_rho = anis_pta.F_mat @ input_map
        # Simulate sig
        if pair_cov:
            inj_pair_cov = np.diag(np.repeat(sim_sig, repeats=anis_pta.npairs))
        else:
            inj_pair_cov = None
        inj_sig = np.repeat(sim_sig, repeats=anis_pta.npairs)

        anis_pta.injected_power = input_map
        anis_pta.injected_rho = inj_rho
        anis_pta.injected_sig = inj_sig
        anis_pta.injected_pair_cov = inj_pair_cov

        if return_vals:
            return input_map, inj_rho, inj_sig, inj_pair_cov


    else:
        raise ValueError("method can only accept 'power_basis' or 'pixel'!")

