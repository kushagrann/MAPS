import numpy as np, sympy as sp, scipy.special as scsp

import pickle, healpy as hp

import numpy.random as nr, scipy.stats as scst
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from enterprise.signals import anis_coefficients as ac

import sympy

import scipy.linalg as sl

from . import clebschGordan as CG

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

def angular_power_spectrum(pta_anis, clm, burn = 4000):
    
    if clm.ndim < 2: 
        maxl = int(np.sqrt(clm.shape[0]))
    else:
        maxl = int(np.sqrt(clm.shape[1]))
        
    if pta_anis.mode == 'power_basis':
        new_clm2 = clm ** 2
    else:
        new_clm2 = np.full(clm.shape, 0.0)
        for ii in range(clm.shape[0]):
            new_clm2[ii] = convert_blm_params_to_clm(pta_anis, clm[ii]) ** 2
            
    if clm.ndim < 2:
        C_l = np.full((maxl), 0.0)
        C_l_err = np.full((maxl), 0.0)
    else:
        C_l = np.full((maxl, new_clm2[burn:, :].shape[0]), 0.0)
    
    idx = 0
    
    for ll in range(maxl):
        
        if ll == 0:
            if clm.ndim < 2:
                C_l[ll] = new_clm2[ll]
                C_l_err[ll] = 2 * np.abs(clm[ll]) * clm_err[ll]
            else:
                C_l[ll] = new_clm2[burn:, ll]
            idx += 1
            
        else:
            subset_len = 2 * ll + 1
            subset = np.arange(idx, idx + subset_len)
            
            if clm.ndim < 2:
                C_l[ll] = np.sum(new_clm2[subset]) / (2 * ll + 1)
                C_l_err[ll] = np.sum(2 * np.abs(clm[subset]) * clm_err[subset]) / (2 * ll + 1)
            else:
                C_l[ll] = np.sum(new_clm2[burn:, subset], axis = 1) / (2 * ll + 1)
            
            idx = subset[-1]
        
    if clm.ndim < 2:
        return C_l, C_l_err
    else:
        return C_l

def draw_random_sample(ip_arr, bins = 50, nsamp = 10):
    
    counts, bin_ed = np.histogram(ip_arr, bins = bins, density = True)
    bin_mid = (bin_ed[1:] + bin_ed[:-1]) / 2.
    
    cdf = np.cumsum(counts) / np.sum(counts)
    
    interp_func = interp1d(cdf, bin_mid)
    
    rn_draw = nr.uniform(low = cdf.min(), high = cdf.max(), size = nsamp)
    
    return interp_func(rn_draw)

def get_posterior_avg_skymap(anis_pta, chain, burn = 5000, n_draws = 10):
    
    burned_chain = chain[burn:, :]
    
    rn_draws = np.full((burned_chain.shape[1] - 4, n_draws), 0.0)
    
    for ii in range(burned_chain.shape[1] - 4):
        
        if ii == 1:
            if anis_pta.mode == 'sqrt_power_basis':
                rn_draws[ii] = np.repeat(1, repeats = n_draws)
            else:
                rn_draws[ii] = np.repeat(np.sqrt(4 * np.pi), repeats = n_draws)
        else:
            rn_draws[ii] = draw_random_sample(burned_chain[:, ii], bins = 20, nsamp = n_draws)
    
    pow_maps = np.full((n_draws, hp.pixelfunc.nside2npix(anis_pta.nside)), 0.0)
    
    for ii in range(n_draws):
        
        if anis_pta.mode == 'sqrt_power_basis':
            clms = convert_blm_params_to_clm(anis_pta, rn_draws[1:, ii])

            pow_maps[ii] = 10 ** rn_draws[0, ii] * ac.mapFromClm(clms, nside = anis_pta.nside)
            
        else:
            
            pow_maps[ii] = 10 ** rn_draws[0, ii] * ac.mapFromClm(rn_draws[1:, ii], nside = anis_pta.nside)
            
    pmean_map = np.mean(pow_maps, axis = 0)
    var_map = np.std(pow_maps, axis = 0)
    
    return pmean_map, var_map
