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

try:
    from scipy.integrate import trapz
except ImportError:
    from scipy.integrate import trapezoid as trapz

from scipy.interpolate import interp1d
from astroML.linear_model import LinearRegression

import tqdm
import random


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


def posterior_sampled_skymap_Cl_orf(anis_pta, data, n_draws=100):

    """ A method to get a posterior sampled skymap, orf and Cl from a chain.
    This method works with all basis.

    Args:
        anis_pta (anis_pta.anis_pta): The anisotropic PTA object
        data (np.ndarray): The posterior samples from the chain file of shape (npars x nsamples)
        n_draws (int): number of draws from chain file. Defaults to 100.

    Returns:
        tuple: A tuple containing:
            pow_map (np.ndarray): (n_draws x n_pixel) numpy array of maps corresponding 
                to n_draws from posterior.
            Cl (np.ndarray): (n_draws x n_Cl) numpy array of C_l values corresponding 
                to n_draws from posterior.
            orf (np.ndarray): (n_draws x n_Cl) numpy array of C_l values corresponding 
                to n_draws from posterior.

    """

    pow_map = np.zeros(shape=(n_draws, hp.pixelfunc.nside2npix(anis_pta.nside)))
    Cl = np.zeros(shape=(n_draws, anis_pta.l_max+1))
    orf = np.zeros(shape=(n_draws, anis_pta.npairs))

    rand_idx = random.sample(population=range(data.shape[1]), k=n_draws)

    if anis_pta.mode == 'hybrid':

        for i,n in enumerate(tqdm.tqdm(rand_idx, desc='n_draw')):
            if anis_pta.activate_A2_pixel:
                pow_map[i] = (10**data[0][n]) * (10**data[1:, n])
                clm_from_map = ac.clmFromMap_fast(h=pow_map[i], lmax=anis_pta.l_max)
                orf[i] = (10**data[0][n]) * anis_pta.orf_from_clm(params=clm_from_map, 
                                                                  include_scale=False)
            else:
                pow_map[i] = 10**data[:, n]
                clm_from_map = ac.clmFromMap_fast(h=pow_map[i], lmax=anis_pta.l_max)
                orf[i] = anis_pta.orf_from_clm(params=clm_from_map, 
                                               include_scale=False)

            Cl[i] = angular_power_spectrum(clm=clm_from_map)

    
    elif anis_pta.mode == 'power_basis':
        
        for i,n in enumerate(tqdm.tqdm(rand_idx, desc='n_draw')):
            pow_map[i] = (10**data[0][n]) * ac.mapFromClm(clm=[np.sqrt(4*np.pi), *data[1:, n]], nside=anis_pta.nside)
            Cl[i] = angular_power_spectrum(clm=np.array([np.sqrt(4*np.pi), *data[1:, n]]))
            orf[i] = (10**data[0][n]) * anis_pta.orf_from_clm(params=np.array([np.sqrt(4*np.pi), *data[1:, n]]), 
                                                              include_scale=False)

    
    elif anis_pta.mode == 'sqrt_power_basis':
        
        for i,n in enumerate(tqdm.tqdm(rand_idx, desc='n_draw')):
            blm_to_clm = convert_blm_params_to_clm(anis_pta, [1.0, *data[1:, n]])
            
            pow_map[i] = (10**data[0][n]) * ac.mapFromClm(clm=blm_to_clm, nside=anis_pta.nside)
            Cl[i] = angular_power_spectrum(clm=blm_to_clm)
            orf[i] = (10**data[0][n]) * anis_pta.orf_from_clm(params=blm_to_clm, 
                                                              include_scale=False)

    
    return pow_map, Cl, orf



def bootstrap_1d(core, param, burn=0, realizations=1000, seed=316):
    """
    A handy function to get boostrapped samples of a parameter's posterior samples.

    Args:
        core (object) : A la_forge.core.Core object of the chain.
        param (str) : The parameter name whose bootstrapped samples to compute.
        burn (int, optional) : No. of samples to be treated as burn-in. Note: This
                                is explicit to burn-in done while creating la_forge.core.Core object.
                                Default is 0.
        realizations (int, optional): No. of realizations for bootstrapping. Default is 1000.

    Returns:
        bootstrapped (np.ndarray) : Bootstrapped samples of 'param' of shape (nrealizations x nsamples).

    """ 
    
    rng = nr.default_rng(seed=seed)  # set up a random number generator
    data = core.get_param(param)[burn:]
    bootstraped = []

    for n in tqdm.tqdm(range(realizations), desc='bootstrapping '+param):
        bootstraped.append(data[rng.choice(len(data), size=len(data), replace=True)])
    
    return np.array(bootstraped)



def get_BF_dist_hypermodel(pta_anis_hypermodel, core, burn=0, realizations=1000, seed=316):
    """
    A handy function to get Bayes' Factor distribution for hypermodel using 'nmodel' parameter.
    This also re-weights the BF if pta_anis.log_weights is not None. Note that this function should
    be used after checking for proper mixing of 'nmodel' from its traceplot.

    Args:
        pta_anis_hypermodel (anis_pta.anis_hypermodel) : The anis_pta hypermodel object used for inference.
        core (object) : A la_forge.core.Core object of the chain.
        burn (int, optional) : No. of samples to be treated as burn-in. Note: This
                                is explicit to burn-in done while creating la_forge.core.Core object.
                                Default is 0.
        realizations (int, optional): No. of realizations for bootstrapping. Default is 1000.

    Returns:
        A dictionary : With key denoting which model/which model. Values corresponding to the BF distribution 
                    including re-weighting of shape (nrealizations).

    """
    
    bootstrapped_nmodel = bootstrap_1d(core, 'nmodel', burn, realizations, seed)

    bf_dist = np.zeros(realizations)
    for r in tqdm.tqdm(range(realizations), desc='Calc. BF dist.'):
        bf_dist[r] = (len(np.where((bootstrapped_nmodel[r, :] > 0.5) & (bootstrapped_nmodel[r, :] <= 1.5))[0]) / 
                       len(np.where((bootstrapped_nmodel[r, :] > -0.5) & (bootstrapped_nmodel[r, :] <= 0.5))[0]))

    if pta_anis_hypermodel.log_weights is not None:
        
        lw_0 = np.exp(float(pta_anis_hypermodel.log_weights[0]))
        lw_1 = np.exp(float(pta_anis_hypermodel.log_weights[1]))
        lw = lw_0 / lw_1
        return {pta_anis_hypermodel.model_names[1]+'/'+pta_anis_hypermodel.model_names[0] : bf_dist * lw}
    
    else:
        return {pta_anis_hypermodel.model_names[1]+'/'+pta_anis_hypermodel.model_names[0] : bf_dist}



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
def inject_anisotropy(anis_pta, method='power_basis', sim_clms=None, sim_blms=None, sim_log10_A2=0.0, sim_sig=0.01, pair_cov=False, seed=42, 
                      h=None, sim_power=50, sim_lon=270, sim_lat=45, sim_pixel_radius=10, include_A2_pixel=False, norm_pixel=False, 
                      add_rand_noise=False, return_vals=False):

    """ A handy function to create a sky with injected anisotropy. 
    This function upon completion creates 'injected' instances of the anis_pta object.

    Args:
        anis_pta (object) : The anis_pta instance created by MAPS.
        method (str, optional): The way of creating the inject sky. 'power_basis' or 'pixel'. 
            Defaults to 'power_basis'.
        sim_clms (list or np.ndarray): The list or array of clm values to inject including c_00.
        sim_blms (list or np.ndarray): The list or array of blm values to inject with amplitude and phase seperated including b_00.
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
        add_rand_noise (bool, optional): Whether to add random gaussian noise in the pixel method.
        return_vals (bool, optional): Whether to return a tuple of injected parameters/power, rho and sig. 
            Defaults to None.

    Returns:
        tuple (optional): A tuple of 3 np.ndarrays containing the 
            injected cross-correlations, injected cross-correlation uncertainties and 
            injected pair covariance matrix (None if pair_cov=False).
            NOTE: The function also creates 'injected' instances of the anis_pta.

    Raises:
        ValueError: If sim_clms not specified when method='power_basis'.
        ValueError: If sim_clms not specified when method='power_basis'.
        ValueError: If method not in ['pixel', 'power_basis', 'sqrt_power_basis'].

    """

    if method == 'pixel':

        if h is not None:
            input_map = h if type(h) is np.ndarray else np.array(h)
            
        else:
            # From MAPS example
            # Simulate an isotropic background plus a hotspot
            input_map = np.ones(anis_pta.npix)
            vec = hp.ang2vec(sim_lon, sim_lat, lonlat=True)
            radius = np.radians(sim_pixel_radius)

            disk_anis = hp.query_disc(nside = anis_pta.nside, vec = vec, radius = radius, inclusive = False)
    
            input_map[disk_anis] += sim_power

        if norm_pixel:
            pixel_area = (4*np.pi) / anis_pta.npix
            input_map = (input_map / trapz(input_map, dx=pixel_area)) * (4*np.pi)

        # Simulate rho
        if include_A2_pixel:
            sim_orf = (10**sim_log10_A2) * (anis_pta.F_mat @ input_map)
        else:
            sim_orf = anis_pta.F_mat @ input_map

        input_clms = ac.clmFromMap_fast(h=input_map, lmax=anis_pta.l_max)
            

    elif method == 'power_basis':

        # If sim_clm not specified
        if sim_clms is None:
            raise ValueError("Specify clm values in sim_clms to do a power_basis injection!")

        input_clms = sim_clms if type(sim_clms) is np.ndarray else np.array(sim_clms)
        sim_orf = (10**sim_log10_A2) * (anis_pta.Gamma_lm.T @ input_clms)

        input_map = (10**sim_log10_A2) * ac.mapFromClm(input_clms, nside=anis_pta.nside)


    elif method == 'sqrt_power_basis':

        # If sim_clm not specified
        if sim_blms is None:
            raise ValueError("Specify blm values in sim_clms to do a sqrt_power_basis injection!")        
        
        input_blms = sim_blms if type(sim_blms) is np.ndarray else np.array(sim_blms)
        ### Convert blm amp & phase to complex blms (still no '-m'; size:l>=1,m>=0->l + 00) b_00 is set internally here
        ### Convert complex blms to alms / complex clms (now with '-m'; size:(lmax+1)**2)
        ### Convert complex clms / alms to real clms and normalize to c_00=root(4pi)
        input_clms = convert_blm_params_to_clm(anis_pta, input_blms) # need to pass b_00 here
        sim_orf = (10**sim_log10_A2) * (anis_pta.Gamma_lm.T @ input_clms)#[:, np.newaxis]) # (ncc x nclm) @ (nclm x 1) => RP - (ncc x 1)

        input_map = (10**sim_log10_A2) * ac.mapFromClm(input_clms, nside=anis_pta.nside)


    else:
        raise ValueError("method can only accept 'pixel', 'power_basis' or 'sqrt_power_basis'!")


    # Simulate sig
    inj_sig = np.repeat(sim_sig, repeats=anis_pta.npairs)
        
    # Add random noise if specified
    if add_rand_noise:
        # Simulate rho - Shift by mean and scale by std
        rng = nr.default_rng(seed=seed)
        normal_dist = rng.normal(size=anis_pta.npairs)
        inj_rho = sim_orf.reshape(anis_pta.npairs) + sim_sig*normal_dist
    else:
        inj_rho = sim_orf

    if pair_cov:
        inj_pair_cov = np.diag(np.repeat(sim_sig, repeats=anis_pta.npairs))
    else:
        inj_pair_cov = None

    
    anis_pta.injected_rho = inj_rho
    anis_pta.injected_sig = inj_sig
    anis_pta.injected_pair_cov = inj_pair_cov

    anis_pta.injected_clms = input_clms
    anis_pta.injected_power = input_map
    anis_pta.injected_Cl = angular_power_spectrum(clm=anis_pta.injected_clms)
    
    
    if return_vals:
        return inj_rho, inj_sig, inj_pair_cov
    

