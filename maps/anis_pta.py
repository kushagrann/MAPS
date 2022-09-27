import numpy as np, sympy as sp, scipy.special as scsp
import scipy.optimize as sopt

import pickle, healpy as hp

import numpy.random as nr, scipy.stats as scst
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from enterprise.signals import anis_coefficients as ac

import sympy

import scipy.linalg as sl

from . import clebschGordan as CG, utils

from scipy.interpolate import interp1d
from astroML.linear_model import LinearRegression

import lmfit
from lmfit import minimize, Parameters

class anis_pta():

    def __init__(self, psrs_theta, psrs_phi, xi = [], rho = [], sig = [], os = 1, l_max = 6, nside = 2,
                mode = 'power_basis', use_physical_prior = False, include_pta_monopole = False):

        self.psrs_theta = psrs_theta
        self.psrs_phi = psrs_phi
        if len(xi) != 0:
            self.xi = xi
        else:
            self.xi = self._get_xi()

        #Read in OS and normalize cross-correlations by OS.
        self.os = os
        if len(rho) > 0 and len(sig) > 0:
            self.rho = np.array(rho) / self.os
            self.sig = np.array(sig) / self.os
        else:
            self.rho = rho
            self.sig = sig

        self.l_max = l_max
        self.nside = nside
        self.use_physical_prior = use_physical_prior
        self.include_pta_monopole = include_pta_monopole

        self.npsrs = len(self.psrs_theta)
        self.npairs = int(np.math.factorial(self.npsrs) / (np.math.factorial(2) * np.math.factorial(self.npsrs - 2)))

        self.npix = hp.nside2npix(self.nside)
        self.gw_theta, self.gw_phi = hp.pix2ang(nside = self.nside, ipix = np.arange(self.npix))

        self.mode = mode

        #Some configuration for spherical harmonic basis runs
        #clm refers to normal spherical harmonic basis
        #blm refers to sqrt power spherical harmonic basis
        self.blmax = int(self.l_max / 2.)
        self.clm_size = (self.l_max + 1) ** 2
        self.blm_size = hp.Alm.getsize(self.blmax)

        self.sqrt_basis_helper = CG.clebschGordan(l_max = self.l_max)
        #self.reorder, self.neg_idx, self.zero_idx, self.pos_idx = self.reorder_hp_ylm()

        if self.mode == 'power_basis':
            self.ndim = 1 + (self.l_max + 1) ** 2
        elif self.mode == 'sqrt_power_basis':
            #self.ndim = 1 + (2 * (hp.Alm.getsize(int(self.blmax)) - self.blmax))
            self.ndim = 1 + (self.blmax + 1) ** 2

        self.F_mat = self.antenna_response()

        if self.mode == 'power_basis' or self.mode == 'sqrt_power_basis':

            #The spherical harmonic basis for \Gamma_lm
            #shape (nclm, npsrs, npsrs)
            Gamma_lm = ac.anis_basis(np.dstack((self.psrs_phi, self.psrs_theta))[0], lmax = self.l_max,
                                     nside = self.nside)

            uti = np.triu_indices(n = self.npsrs, k = 1)

            self.Gamma_lm = np.full((Gamma_lm.shape[0], self.rho.shape[0]), 0.0)

            for ii in range(Gamma_lm.shape[0]):

                self.Gamma_lm[ii] = Gamma_lm[ii][uti]

        return None

    def _get_radec(self):

        psr_ra = []
        psr_dec = []

        for ptheta, pphi in zip(self.psrs_theta, self.psrs_phi):

            psr_ra.append(pphi)
            psr_dec.append(np.pi / 2 - ptheta)

        return psr_ra, psr_dec

    def _get_xi(self):

        psrs_ra, psrs_dec = self._get_radec()

        pos_vectors = np.array(
                [np.cos(psrs_ra) * np.cos(psrs_dec), np.sin(psrs_ra) * np.cos(psrs_dec), np.sin(psrs_dec)])

        xi = []

        for ii in range(len(self.psrs_theta)):
                for jj in range(ii+1, len(self.psrs_theta)):

                    xi.append(np.arccos(np.dot(pos_vectors[:, ii], pos_vectors[:, jj])))

        return np.array(xi)

    def _fplus_fcross(self, psrtheta, psrphi, gwtheta, gwphi):
        """
        Compute gravitational-wave quadrupolar antenna pattern.
        (From NX01)
        :param psr: pulsar object
        :param gwtheta: Polar angle of GW source in celestial coords [radians]
        :param gwphi: Azimuthal angle of GW source in celestial coords [radians]
        :returns: fplus, fcross
        """

        # define variable for later use
        cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
        singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)

        # unit vectors to GW source
        m = np.array([singwphi, -cosgwphi, 0.0])
        n = np.array([-cosgwtheta*cosgwphi, -cosgwtheta*singwphi, singwtheta])
        omhat = np.array([-singwtheta*cosgwphi, -singwtheta*singwphi, -cosgwtheta])

        # pulsar location
        #ptheta = np.pi/2 - psr.decj
        #pphi = psr.raj
        ptheta = psrtheta
        pphi = psrphi

        # use definition from Sesana et al 2010 and Ellis et al 2012
        phat = np.array([np.sin(ptheta)*np.cos(pphi), np.sin(ptheta)*np.sin(pphi),\
                np.cos(ptheta)])

        fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1 - np.dot(omhat, phat))
        fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 - np.dot(omhat, phat))

        return fplus, fcross

    def antenna_response(self):

        F_mat = np.zeros((self.npairs, self.npix))
        pair_no = 0

        for ii in range(self.npsrs):

            for jj in range(ii + 1, self.npsrs):

                for kk in range(self.npix):

                    pp_1, pc_1 = self._fplus_fcross(self.psrs_theta[ii], self.psrs_phi[ii],
                                              self.gw_theta[kk], self.gw_phi[kk])
                    pp_2, pc_2 = self._fplus_fcross(self.psrs_theta[jj], self.psrs_phi[jj],
                                              self.gw_theta[kk], self.gw_phi[kk])

                    F_mat[pair_no][kk] =  (pp_1 * pp_2 + pc_1 * pc_2)  * 1.5 / (self.npix)

                pair_no += 1

        return F_mat

    def get_pure_HD(self):
        #Return the theoretical HD curve given xi

        xx = (1 - np.cos(self.xi)) / 2.
        hd_curve = 1.5 * xx * np.log(xx) - xx / 4 + 0.5

        return hd_curve

    def orf_from_clm(self, params):
        #Using supplied clm values, calculate the corresponding power map
        #and calculate the ORF from that power map (convoluted, I know)

        amp2 = 10 ** params[0]
        clm = params[1:]

        sh_map = ac.mapFromClm(clm, nside = self.nside)

        orf = amp2 * np.dot(self.F_mat, sh_map)

        return np.longdouble(orf)

    def clmFromAlm(self, alm):
        """
        Given an array of clm values, return an array of complex alm valuex
        Note: There is a bug in healpy for the negative m values. This function
        just takes the imaginary part of the abs(m) alm index.
        """
        nalm = len(alm)
        #maxl = int(np.sqrt(9.0 - 4.0 * (2.0 - 2.0 * nalm)) * 0.5 - 1.5)  # Really?
        maxl = self.l_max
        nclm = (maxl + 1) ** 2

         #Check the solution. Went wrong one time..
        #if nalm != int(0.5 * (maxl + 1) * (maxl + 2)):
        #    raise ValueError("Check numerical precision. This should not happen")

        clm = np.zeros(nclm)

        clmindex = 0
        for ll in range(0, maxl + 1):
            for mm in range(-ll, ll + 1):
                almindex = hp.Alm.getidx(maxl, ll, abs(mm))

                if mm == 0:
                    clm[clmindex] = alm[almindex].real
                elif mm < 0:
                    clm[clmindex] = (-1) ** mm * alm[almindex].imag * np.sqrt(2)
                elif mm > 0:
                    clm[clmindex] = (-1) ** mm * alm[almindex].real * np.sqrt(2)

                clmindex += 1

        return clm

    def max_lkl_pixel(self, cutoff = 0, return_fac1 = False, use_svd_reg = False, reg_type = 'l2', alpha = 0):

        N_mat = np.zeros((len(self.rho), len(self.rho)))
        N_mat_inv = np.zeros((len(self.rho), len(self.rho)))

        N_mat[np.diag_indices(N_mat.shape[0])] = self.sig ** 2
        N_mat_inv[np.diag_indices(N_mat_inv.shape[0])] = 1 / self.sig ** 2

        sv = sl.svd(np.matmul(self.F_mat.transpose(), np.matmul(N_mat_inv, self.F_mat)), compute_uv = False)

        cn = np.max(sv) / np.min(sv)

        if use_svd_reg:

            abs_cutoff = cutoff * np.max(sv)

            fac1 = sl.pinvh(np.matmul(self.F_mat.transpose(), np.matmul(N_mat_inv, self.F_mat)), cond = abs_cutoff)

            fac2 = np.matmul(self.F_mat.transpose(), np.matmul(N_mat_inv, self.rho))

            pow_err = np.sqrt(np.diag(fac1))

            power = np.matmul(fac1, fac2)

        else:

            diag_identity = np.diag(np.full(self.F_mat.shape[1], 1))

            fac1r = sl.pinvh(np.matmul(self.F_mat.transpose(), np.matmul(N_mat_inv, self.F_mat)) + alpha * diag_identity)

            pow_err = np.sqrt(np.diag(fac1r))

            clf = LinearRegression(regularization = reg_type, fit_intercept = False, kwds = dict(alpha = alpha))

            clf.fit(self.F_mat, self.rho, self.sig)

            power = clf.coef_

        return power, pow_err, cn, sv

    def fisher_matrix_sph(self):

        N_mat = np.zeros((len(self.rho), len(self.rho)))
        N_mat_inv = np.zeros((len(self.rho), len(self.rho)))

        N_mat[np.diag_indices(N_mat.shape[0])] = self.sig ** 2
        N_mat_inv[np.diag_indices(N_mat_inv.shape[0])] = 1 / self.sig ** 2

        F_mat_clm = self.Gamma_lm.transpose()

        fisher_mat = np.matmul(F_mat_clm.transpose(), np.matmul(N_mat_inv, F_mat_clm))

        return fisher_mat

    def fisher_matrix_pixel(self):

        N_mat = np.zeros((len(self.rho), len(self.rho)))
        N_mat_inv = np.zeros((len(self.rho), len(self.rho)))

        N_mat[np.diag_indices(N_mat.shape[0])] = self.sig ** 2
        N_mat_inv[np.diag_indices(N_mat_inv.shape[0])] = 1 / self.sig ** 2

        fisher_mat = np.matmul(self.F_mat.transpose(), np.matmul(N_mat_inv, self.F_mat))

        return fisher_mat

    def max_lkl_clm(self, cutoff = 0, use_svd_reg = False, reg_type = 'l2', alpha = 0):

        N_mat = np.zeros((len(self.rho), len(self.rho)))
        N_mat_inv = np.zeros((len(self.rho), len(self.rho)))

        N_mat[np.diag_indices(N_mat.shape[0])] = self.sig ** 2
        N_mat_inv[np.diag_indices(N_mat_inv.shape[0])] = 1 / self.sig ** 2

        F_mat_clm = self.Gamma_lm.transpose()

        sv = sl.svd(np.matmul(F_mat_clm.transpose(), np.matmul(N_mat_inv, F_mat_clm)), compute_uv = False)

        cn = np.max(sv) / np.min(sv)

        if use_svd_reg:

            abs_cutoff = cutoff * np.max(sv)

            fac1 = sl.pinvh(np.matmul(F_mat_clm.transpose(), np.matmul(N_mat_inv, F_mat_clm)), cond = abs_cutoff)

            fac2 = np.matmul(F_mat_clm.transpose(), np.matmul(N_mat_inv, self.rho))

            clms = np.matmul(fac1, fac2)

            clm_err = np.sqrt(np.diag(fac1))

        else:

            diag_identity = np.diag(np.full(F_mat_clm.shape[1], 1.0))

            fac1r = sl.pinvh(np.matmul(F_mat_clm.transpose(), np.matmul(N_mat_inv, F_mat_clm)) + alpha * diag_identity)

            clf = LinearRegression(regularization = reg_type, fit_intercept = False, kwds = dict(alpha = alpha))

            clf.fit(F_mat_clm, self.rho, self.sig)

            clms = clf.coef_

            clm_err = np.sqrt(np.diag(fac1r))

        return clms, clm_err, cn, sv

    def setup_lmfit_parameters(self):

        #This shape is to fit with lmfit's Parameter.add_many();
        #Format is (name, value, vary, min, max, expr, brute_step)
        if self.include_pta_monopole:
            x0 = np.full((self.ndim + 1, 7), 0.0, dtype = object)
        else:
            x0 = np.full((self.ndim, 7), 0.0, dtype = object)

        if self.include_pta_monopole:
            x0[0] = np.array(['A_mono', np.log10(nr.uniform(1e-2, 3)), True, np.log10(1e-2), np.log10(1e2), None, None])
            x0[1] = np.array(['A2', np.log10(nr.uniform(1e-2, 3)), True, np.log10(1e-2), np.log10(1e2), None, None])

        else:
            x0[0] = np.array(['A2', np.log10(nr.uniform(0, 3)), True, np.log10(1e-2), np.log10(1e2), None, None])

        if self.include_pta_monopole:
            idx = 2
        else:
            idx = 1

        for ll in range(self.blmax + 1):
            for mm in range(0, ll + 1):

                if ll == 0:
                    x0[idx] = np.array(['b_{}{}'.format(ll, mm), 1., False, None, None, None, None])
                    idx += 1

                elif mm == 0:
                    x0[idx] = 0 #nr.uniform(0, 5)
                    x0[idx] = np.array(['b_{}{}'.format(ll, mm), nr.uniform(-1, 1), True, None, None, None, None])
                    idx += 1

                elif mm != 0:
                    #x0[idx] = 0 #nr.uniform(0, 5)
                    #Amplitude is always >= 0; initial guess set to small non-zero value
                    x0[idx] = np.array(['b_{}{}_amp'.format(ll, mm), nr.uniform(0, 3), True, 0, None, None, None])
                    #x0[idx + 1] = 0 #nr.uniform(0, 2 * np.pi)
                    x0[idx + 1] = np.array(['b_{}{}_phase'.format(ll, mm), nr.uniform(0, 2 * np.pi), True, 0, 2 * np.pi, None, None])
                    idx += 2

        #Convert the numpy array to tuple of tuples to use with lmfit Parameter
        x0_tuple = tuple(map(tuple, x0))

        #Setup the Parameters() object
        params = Parameters()

        params.add_many(*x0_tuple)

        return params

    def max_lkl_sqrt_power(self, params = np.array(())):

        if len(params) == 0:
            params = self.setup_lmfit_parameters()
        else:
            params = params

        def residuals(params, obs_orf, obs_orf_err):

            #Get the input parameters as a dictionary
            param_dict = params.valuesdict()

            #Conver the values to a numpy array for using in other pta_anis functions
            param_arr = np.array(list(param_dict.values()))

            #Do the thing
            if self.include_pta_monopole:
                clm_pred = utils.convert_blm_params_to_clm(self, param_arr[2:])
            else:
                clm_pred = utils.convert_blm_params_to_clm(self, param_arr[1:])

            if self.include_pta_monopole:
                sim_orf = (10 ** param_arr[1]) * np.sum(clm_pred[:, np.newaxis] * self.Gamma_lm, axis = 0) + (10 ** param_arr[0]) * 1
            else:
                sim_orf = (10 ** param_arr[0]) * np.sum(clm_pred[:, np.newaxis] * self.Gamma_lm, axis = 0)

            return (sim_orf - obs_orf) / obs_orf_err

        #Get initial fit from function above. Includes initial guess
        #init_guess = self.setup_lmfit_parameters()

        #Setup lmfit minimizer and get solution
        mini = lmfit.Minimizer(residuals, params, fcn_args=(self.rho, self.sig))
        opt_params = mini.minimize()

        #Return the full output object for user.
        #Post-processing help in utils and lmfit documentation
        return opt_params

    def prior(self, params):

        if self.mode == 'power_basis':
            amp2 = params[0]
            clm = params[1:]
            maxl = int(np.sqrt(len(clm)))

            if clm[0] != np.sqrt(4 * np.pi) or any(np.abs(clm[1:]) > 15) or (amp2 < np.log10(1e-5) or amp2 > np.log10(1e3)):
                return -np.inf #0 #np.longdouble(1e-300)
            elif self.use_physical_prior:
                #NOTE: if using physical prior, make sure to set initial sample to isotropy

                sh_map = ac.mapFromClm(clm, nside = self.nside)

                if np.any(sh_map < 0):
                    return -np.inf #0
                else:
                    return np.longdouble((1 / 10) ** (len(params) - 1))

            else:
                return np.longdouble((1 / 10) ** (len(params) - 1))

        elif self.mode == 'sqrt_power_basis':
            amp2 = params[0]
            blm_params = params[1:]

            if (amp2 < np.log10(1e-5) or amp2 > np.log10(1e3)):
                return -np.inf #0

            else:
                idx = 1
                for ll in range(self.blmax + 1):
                    for mm in range(0, ll + 1):

                        if ll == 0 and params[idx] != 1:
                            return -np.inf #0

                        if mm == 0 and (params[idx] > 5 or params[idx] < -5):
                            return -np.inf #0

                        if mm != 0 and (params[idx] > 5 or params[idx] < 0 or params[idx + 1] >= 2 * np.pi or params[idx + 1] < 0):
                            return -np.inf #0

                        if ll == 0:
                            idx += 1
                        elif mm == 0:
                            idx += 1
                        else:
                            idx += 2

                return np.longdouble((1 / 10) ** (len(params) - 1))

    def logLikelihood(self, params):

        if self.mode == 'hybrid':

            amp2 = params[0]
            clm = params[1:]

            #Fix c_00 to sqrt(4 * pi)
            clm[0] = np.sqrt(4 * np.pi)

            sim_orf = self.orf_from_clm(params)

            #Decompose the likelihood which is a product of gaussians into sums when getting log-likely
            alpha = (self.rho - sim_orf) ** 2 / (2 * self.sig ** 2)
            beta = np.longdouble(1 / (self.sig * np.sqrt(2 * np.pi)))

            loglike = np.sum(np.log(beta)) - np.sum(alpha)

        elif self.mode == 'power_basis':

            amp2 = 10 ** params[0]
            clm = params[1:]

            #clm[0] = np.sqrt(4 * np.pi)

            sim_orf = amp2 * np.sum(clm[:, np.newaxis] * self.Gamma_lm, axis = 0)

            alpha = (self.rho - sim_orf) ** 2 / (2 * self.sig ** 2)
            beta = np.longdouble(1 / (self.sig * np.sqrt(2 * np.pi)))

            loglike = np.sum(np.log(beta)) - np.sum(alpha)

        elif self.mode == 'sqrt_power_basis':

            amp2 = 10 ** params[0]
            blm_params = params[1:]

            #blm_params[0] = 1

            #b00 is set internally, no need to pass it in
            blm = self.sqrt_basis_helper.blm_params_2_blms(blm_params[1:])

            new_clms = self.sqrt_basis_helper.blm_2_alm(blm)

            #Only need m >= 0 entries for clmfromalm
            clms_rvylm = self.clmFromAlm(new_clms)
            #clms_rvylm[0] *= 6.283185346689728

            sim_orf = amp2 * np.sum(clms_rvylm[:, np.newaxis] * self.Gamma_lm, axis = 0)

            alpha = (self.rho - sim_orf) ** 2 / (2 * self.sig ** 2)
            beta = np.longdouble(1 / (self.sig * np.sqrt(2 * np.pi)))

            loglike = np.sum(np.log(beta)) - np.sum(alpha)

        return loglike
        #return np.prod(gauss, dtype = np.longdouble)

    def logPrior(self, params):
        return np.log(self.prior(params))

    def get_random_sample(self):

        if self.mode == 'power_basis':

            if self.use_physical_prior:
                x0 = np.append(np.append(np.log10(nr.uniform(0, 30, 1)), np.array([np.sqrt(4 * np.pi)])), np.repeat(0, self.ndim - 2))
            else:
                x0 = np.append(np.append(np.log10(nr.uniform(0, 30, 1)), np.array([np.sqrt(4 * np.pi)])), nr.uniform(-5, 5, self.ndim - 2))

        elif self.mode == 'sqrt_power_basis':

            x0 = np.full(self.ndim, 0.0)

            x0[0] = nr.uniform(np.log10(1e-5), np.log10(30))

            idx = 1
            for ll in range(self.blmax + 1):
                for mm in range(0, ll + 1):

                    if ll == 0:
                        x0[idx] = 1.
                        idx += 1

                    elif mm == 0:
                        x0[idx] = 0 #nr.uniform(0, 5)
                        idx += 1

                    elif mm != 0:
                        x0[idx] = 0 #nr.uniform(0, 5)
                        x0[idx + 1] = 0 #nr.uniform(0, 2 * np.pi)
                        idx += 2

        return x0

    def amplitude_scaling_factor(self):
        return 1 / (2 * 6.283185346689728)

class anis_hypermodel():

    def __init__(self, models, log_weights = None, mode = 'sqrt_power_basis', use_physical_prior = True):

        self.models = models
        self.num_models = len(self.models)
        self.mode = mode
        self.use_physical_prior = use_physical_prior

        if log_weights is None:
            self.log_weights = log_weights
        else:
            self.log_weights = np.longdouble(log_weights)

        self.nside = np.max([xx.nside for xx in self.models])

        self.ndim = np.max([xx.ndim for xx in self.models]) + 1

        self.l_max = np.max([xx.l_max for xx in self.models])

        #Some configuration for spherical harmonic basis runs
        #clm refers to normal spherical harmonic basis
        #blm refers to sqrt power spherical harmonic basis
        self.blmax = int(self.l_max / 2)
        self.clm_size = (self.l_max + 1) ** 2
        self.blm_size = hp.Alm.getsize(self.blmax)

    def _standard_prior(self, params):

        if self.mode == 'power_basis':
            amp2 = params[0]
            clm = params[1:]
            maxl = int(np.sqrt(len(clm)))

            if clm[0] != np.sqrt(4 * np.pi) or any(np.abs(clm[1:]) > 15) or (amp2 < np.log10(1e-5) or amp2 > np.log10(1e3)):
                return -np.inf
            elif self.use_physical_prior:
                #NOTE: if using physical prior, make sure to set initial sample to isotropy

                sh_map = ac.mapFromClm(clm, nside = self.nside)

                if np.any(sh_map < 0):
                    return -np.inf
                else:
                    return np.longdouble((1 / 10) ** (len(params) - 1))

            else:
                return np.longdouble((1 / 10) ** (len(params) - 1))

        elif self.mode == 'sqrt_power_basis':
            amp2 = params[0]
            blm_params = params[1:]

            if (amp2 < np.log10(1e-5) or amp2 > np.log10(1e3)):
                return -np.inf

            else:
                idx = 1
                for ll in range(self.blmax + 1):
                    for mm in range(0, ll + 1):

                        if ll == 0 and params[idx] != 1:
                            return -np.inf #0

                        if mm == 0 and (params[idx] > 5 or params[idx] < -5):
                            return -np.inf #0

                        if mm != 0 and (params[idx] > 5 or params[idx] < 0 or params[idx + 1] >= 2 * np.pi or params[idx + 1] < 0):
                            return -np.inf #0

                        if ll == 0:
                            idx += 1
                        elif mm == 0:
                            idx += 1
                        else:
                            idx += 2

                return np.longdouble((1 / 10) ** (len(params) - 1))

    def logPrior(self, params):

        nmodel = int(np.rint(params[0]))
        #print(nmodel)

        if nmodel <= -0.5 or nmodel > 0.5 * (2 * self.num_models - 1):
            return -np.inf

        return np.log(self._standard_prior(params[1:]))

    def logLikelihood(self, params):

        nmodel = int(np.rint(params[0]))
        #print(nmodel)

        active_lnlkl = self.models[nmodel].logLikelihood(params[1: self.models[nmodel].ndim + 1])

        if self.log_weights is not None:
            active_lnlkl += self.log_weights[nmodel]

        return active_lnlkl

    def get_random_sample(self):

        if self.mode == 'power_basis':

            if self.use_physical_prior:
                x0 = np.append(np.append(np.log10(nr.uniform(0, 30, 1)), np.array([np.sqrt(4 * np.pi)])), np.repeat(0, self.ndim - 3))
                x0 = np.append(nr.uniform(-0.5, 0.5 + self.num_models, 1), x0)
            else:
                x0 = np.append(np.append(np.log10(nr.uniform(0, 30, 1)), np.array([np.sqrt(4 * np.pi)])), nr.uniform(-5, 5, self.ndim - 3))
                x0 = np.append(nr.uniform(-0.5, 0.5 + self.num_models, 1), x0)

        elif self.mode == 'sqrt_power_basis':

            x0 = np.full(self.ndim, 0.0)

            x0[0] = nr.uniform(-0.5, 0.5 * (2 * self.num_models - 1), 1)

            x0[1] = nr.uniform(np.log10(1e-5), np.log10(30))

            idx = 2
            for ll in range(self.blmax + 1):
                for mm in range(0, ll + 1):

                    if ll == 0:
                        x0[idx] = 1.
                        idx += 1

                    elif mm == 0:
                        x0[idx] = 0 #nr.uniform(0, 5)
                        idx += 1

                    elif mm != 0:
                        x0[idx] = 0 #nr.uniform(0, 5)
                        x0[idx + 1] = 0 #nr.uniform(0, 2 * np.pi)
                        idx += 2

        return x0
