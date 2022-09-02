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

def convert_blm_params_to_clm(pta_anis, blm_params):

    blm = pta_anis.sqrt_basis_helper.blm_params_2_blms(blm_params[1:])

    #Note that when using this mode, the supplied params need to be in
    #the healpy format

    new_clms = pta_anis.sqrt_basis_helper.blm_2_alm(blm)

    #Only need m >= 0 entries for clmfromalm
    clms_rvylm = pta_anis.clmFromAlm(new_clms)
    #clms_rvylm[0] *= 12.56637061

    return clms_rvylm

def signal_to_noise(pta, lm_params = None):

    if lm_params is None:
        lm_out = pta.max_lkl_sqrt_power()
    else:
        lm_out = lm_params

    pta_mono = ap.anis_pta(pta.psrs_theta, pta.psrs_phi, pta.xi, pta.rho, pta.sig, nside = pta.nside,
                            l_max = 0, mode = pta.mode, os = 1, include_pta_monopole = pta.include_pta_monopole) #OS already applied in pta
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

def angular_power_spectrum(pta_anis, clm):

    maxl = int(np.sqrt(clm.shape[0]))

    new_clm2 = clm ** 2

    C_l = np.full((maxl), 0.0)

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

    counts, bin_ed = np.histogram(ip_arr, bins = bins, density = True)
    bin_mid = (bin_ed[1:] + bin_ed[:-1]) / 2.

    cdf = np.cumsum(counts) / np.sum(counts)

    interp_func = interp1d(cdf, bin_mid)

    rn_draw = nr.uniform(low = cdf.min(), high = cdf.max(), size = nsamp)

    return interp_func(rn_draw)

def posterior_sampled_Cl_skymap(anis_pta, chain, burn = 0, n_draws = 100):
    """
    Return collection of sky maps randomly drawn from posterior
    given an (emcee) chain file sampling the posterior around the
    maximum likelihood value

    Parameters
    --------------
    anis_pta --> the anisotropic PTA object
    chain --> chain file with posterior samples; expected to be a pd.DataFrame with rows = samples, params = columns
    burn --> burn-in length (optional)
    n_draws --> number of draws from chain file to generate the posterior averaged sky map

    Returns
    --------------
    pow_map --> (n_draws x n_pixel) numpy array of maps corresponding to n_draws from posterior.
    Cl --> (n_draws x n_Cl) numpy array of C_l values corresponding to n_draws from posterior,
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
