import numpy as np, sympy as sp, scipy.special as scsp
import scipy.optimize as sopt

import os
import pickle, healpy as hp

import numpy.random as nr, scipy.stats as scst
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise.signals import parameter

from enterprise.signals import anis_coefficients as ac

import sympy

try:
    from scipy.integrate import trapz
except ImportError:
    from scipy.integrate import trapezoid as trapz

import scipy.linalg as sl

### Correct back to .
from maps import clebschGordan as CG, utils

from scipy.interpolate import interp1d
from astroML.linear_model import LinearRegression

import lmfit
from lmfit import minimize, Parameters

# Bilby packages to import
import bilby
from bilby.core.prior import Uniform

class anis_pta():
    """A class to perform anisotropic GW searches using PTA data.

    This class can be used to perform anisotropic GW searches using PTA data by supplying
    it with the outputs of the PTA optimal statistic which can be found in
    (enterprise_extensions.frequentist.optimal_statistic or defiant/optimal_statistic).
    While you can include the OS values upon construction (xi, rho, sig, os),
    you can also use the method set_data() to set these values after construction.

    Attributes:
        psrs_theta (np.ndarray): An array of pulsar position theta [npsr].
        psrs_phi (np.ndarray): An array of pulsar position phi [npsr].
        npsr (int): The number of pulsars in the PTA.
        npair (int): The number of pulsar pairs.
        pair_idx (np.ndarray): An array of pulsar indices for each pair [npair x 2].
        xi (np.ndarray, optional): A list of pulsar pair separations from the OS [npair].
        rho (np.ndarray, optional): A list of pulsar pair correlations [npair].
            NOTE: rho is normalized by the OS value, making this slightly different from 
            what the OS uses. (i.e. OS calculates \hat{A}^2 * ORF while this uses ORF).
        sig (np.ndarray, optional): A list of 1-sigma uncertainties on rho [npair].
        os (float, optional): The optimal statistic's best-fit A^2 value.
        pair_cov (np.ndarray, optional): The pair covariance matrix [npair x npair].
        pair_ind_N_inv (np.ndarray): The inverse of the pair independent covariance matrix.
        pair_cov_N_inv (np.ndarray): The inverse of the pair covariance matrix.
        l_max (int): The maximum l value for spherical harmonics.
        nside (int): The nside of the healpix sky pixelization.
        npix (int): The number of pixels in the healpix pixelization.
        blmax (int): The maximum l value for the sqrt power basis.
        clm_size (int): The number of spherical harmonic modes.
        blm_size (int): The number of spherical harmonic modes for the sqrt power basis.
        gw_theta (np.ndarray): An array of source GW theta positions [npix].
        gw_phi (np.ndarray): An array of source GW phi positions [npix].
        use_physical_prior (bool): Whether to use physical priors or not.
        include_pta_monopole (bool): Whether to include the monopole term in the search.
        mode (str): The mode of the spherical harmonic decomposition to use.
            Must be 'power_basis', 'sqrt_power_basis', or 'hybrid'.
        sqrt_basis_helper (CG.clebschGordan): A helper object for the sqrt power basis.
        ndim (int): The number of dimensions for the search.
        F_mat (np.ndarray): The antenna response matrix [npair x npix].
        Gamma_lm (np.ndarray): The spherical harmonic basis [npair x ndim].
    """

    def __init__(self, psrs_theta, psrs_phi, xi = None, rho = None, sig = None, 
                 os = None, pair_cov = None, l_max = 6, nside = 2, mode = 'power_basis', 
                 use_physical_prior = False, include_pta_monopole = False, 
                 pair_idx = None):
        """Constructor for the anis_pta class.

        This function will construct an instance of the anis_pta class. This class
        can be used to perform anisotropic GW searches using PTA data by supplying
        it with the outputs of the PTA optimal statistic which can be found in 
        (enterprise_extensions.frequentist.optimal_statistic or defiant/optimal_statistic).
        While you can include the OS values upon construction (xi, rho, sig, os),
        you can also use the method set_data() to set these values after construction.
        
        Args:
            psrs_theta (np.ndarray): An array of pulsar position theta [npsr].
            psrs_phi (np.ndarray): An array of pulsar position phi [npsr].
            xi (np.ndarray, optional): A list of pulsar pair separations from the OS [npair].
            rho (np.ndarray, optional): A list of pulsar pair correlations [npair].
            sig (np.ndarray, optional): A list of 1-sigma uncertainties on rho [npair].
            os (float, optional): The optimal statistic's best-fit A^2 value.
            pair_cov (np.ndarray, optional): The pair covariance matrix [npair x npair].
            l_max (int): The maximum l value for spherical harmonics.
            nside (int): The nside of the healpix sky pixelization.
            mode (str): The mode of the spherical harmonic decomposition to use.
                Must be 'power_basis', 'sqrt_power_basis', or 'hybrid'.
            use_physical_prior (bool): Whether to use physical priors or not.
            include_pta_monopole (bool): Whether to include the monopole term in the search.
            pair_idx (np.ndarray, optional): An array of pulsar indices for each pair [npair x 2].

        Raises:
            ValueError: If the lengths of psrs_theta and psrs_phi are not equal.
            ValueError: If the length of pair_idx is not equal to the number of pulsar pairs.            
        """
        # Pulsar positions
        self.psrs_theta = psrs_theta if type(psrs_theta) is np.ndarray else np.array(psrs_theta)
        self.psrs_phi = psrs_phi if type(psrs_phi) is np.ndarray else np.array(psrs_phi)
        if len(psrs_theta) != len(psrs_phi):
            raise ValueError("Pulsar theta and phi arrays must have the same length")
        self.npsr = len(psrs_theta)
        self.npairs = int( (self.npsr * (self.npsr - 1)) / 2)

        # OS values
        if pair_idx is None:
            self.pair_idx = np.array([(a,b) for a in range(self.npsr) for b in range(a+1,self.npsr)])
        else:
            self.pair_idx = pair_idx
        
        if xi is not None:
            if type(xi) is not np.ndarray:
                self.xi = np.array(xi)
            else:
                self.xi = xi
        else:
            self.xi = self._get_xi()

        self.rho, self.sig, self.os, self.pair_cov = None, None, None, None
        self.pair_ind_N_inv, self.pair_cov_N_inv = None, None

        self.set_data(rho, sig, os, pair_cov)
        
        # Check if pair_idx is valid
        if len(self.pair_idx) != self.npairs:
            raise ValueError("pair_idx must have length equal to the number of pulsar pairs")
        
        # Pixel decomposition and Spherical harmonic parameters
        self.l_max = int(l_max)
        self.nside = int(nside)
        self.npix = hp.nside2npix(self.nside)

        # Some configuration for spherical harmonic basis runs
        # clm refers to normal spherical harmonic basis
        # blm refers to sqrt power spherical harmonic basis
        self.blmax = int(self.l_max / 2.)
        self.clm_size = (self.l_max + 1) ** 2
        self.blm_size = hp.Alm.getsize(self.blmax)

        self.gw_theta, self.gw_phi = hp.pix2ang(nside=self.nside, ipix=np.arange(self.npix))
       
        self.use_physical_prior = bool(use_physical_prior)
        self.include_pta_monopole = bool(include_pta_monopole)

        if mode in ['power_basis', 'sqrt_power_basis', 'hybrid']:
            self.mode = mode
        else:
            raise ValueError("mode must be either 'power_basis','sqrt_power_basis' or 'hybrid'")

        
        self.sqrt_basis_helper = CG.clebschGordan(l_max = self.l_max)
        #self.reorder, self.neg_idx, self.zero_idx, self.pos_idx = self.reorder_hp_ylm()

        if self.mode == 'power_basis':
            self.ndim = 1 + (self.l_max + 1) ** 2
        elif self.mode == 'sqrt_power_basis':
            #self.ndim = 1 + (2 * (hp.Alm.getsize(int(self.blmax)) - self.blmax))
            self.ndim = 1 + (self.blmax + 1) ** 2

        self.F_mat = self.antenna_response()

        if self.mode == 'power_basis' or self.mode == 'sqrt_power_basis':

            # The spherical harmonic basis for \Gamma_lm_mat shape (nclm, npsrs, npsrs)
            Gamma_lm_mat = ac.anis_basis(np.dstack((self.psrs_phi, self.psrs_theta))[0], 
                                         lmax = self.l_max, nside = self.nside)
            
            # We need to reorder Gamma_lm_mat to shape (nclm, npairs)
            self.Gamma_lm = np.zeros((Gamma_lm_mat.shape[0], self.npairs))
            for i, (a, b) in enumerate(self.pair_idx):
                self.Gamma_lm[:, i] = Gamma_lm_mat[:, a, b]

        return None
    
    
    def set_data(self, rho=None, sig=None, os=None, covariance=None):
        """Set the data for the anis_pta object.

        This function allows you to set the data for the anis_pta object 
        after construction. This allows users to use the same anis_pta object
        with different draws of the data. This is especially helpful when combined
        with the noise marginalized optimal statistic or per-frequency optimal statistic 
        analyses. This function will normalize the rho, sig, and covariance by the 
        OS (A^2) value, making self.rho, self.sig, and self.pair_cov represent only 
        the correlations. 
        NOTE: If using pair covariance you still need to supply this function
        with the pairwise uncertainties as well!

        Args:
            rho (list, optional): A list of pulsar pair correlated amplitudes (<rho> = <A^2*ORF>).
            sig (list, optional): A list of 1-sigma uncertaintties on rho.
            os (float, optional): The OS' fit A^2 value.
            covariance (np.ndarray, optional): The pair covariance matrix [npair x npair].
        """
        # Read in OS and normalize cross-correlations by OS. 
        # (i.e. get <rho/OS> = <ORF>)
        self._Lt_pc, self._Lt_nopc = None, None # Reset the cholesky decompositions

        if (rho is not None) and (sig is not None) and (os is not None):
            self.os = os
            self.rho = np.array(rho) / self.os
            self.sig = np.array(sig) / self.os

            # Set the inverse of the pair independent covariance matrix
            self.pair_ind_N_inv = self._get_N_inv(pair_cov = False)

        else:
            self.rho = None
            self.sig = None
            self.os = None

        if covariance is not None:
            self.pair_cov = covariance / self.os**2
            
            # A handy attribute to be used in likelihood evaluation
            self._lik_denom = np.linalg.slogdet(2 * np.pi * self.pair_cov)[1]

            # Get the inverse of the pair covariance matrix
            self.pair_cov_N_inv = self._get_N_inv(pair_cov = True)

        else:
            self.pair_cov = None
            self.pair_cov_N_inv = None


    def _get_N_inv(self, pair_cov=False, ret_cond = False):
        """A method to calculate the inverse of the pair covariance matrix N.

        This function will calculate the inverse of the pair covariance matrix N.
        If pair_cov is False, it will use a diagonal matrix consisting of the 
        squared sig values. 
        If pair_cov is True, it will use the woodbury identity to calculate the
        inverse of N with the pair covariance matrix to increase the stability of 
        the inverse.

        Args:
            pair_cov (bool): A flag to use the pair covariance matrix. Defaults to False.
            ret_cond (bool, optional): A flag to return the condition number of the 
                covariance matrix. Only useful when using pair covariance.

        Returns:
            np.ndarray or tuple: The inverse of the pair covariance matrix N. If ret_cond
                is True, it will return a tuple containing the inverse and the condition
                number of the covariance matrix.
        """
        if pair_cov and self.pair_cov is None:
            raise ValueError("No pair covariance matrix supplied! Set it with set_data()")
        elif pair_cov:
            A = np.diag(self.sig ** 2)
            K = self.pair_cov - A
            In = np.eye(A.shape[0])

            N_inv, cond = utils.woodbury_inverse(A, In, In, K, ret_cond = True)

            if ret_cond:
                return N_inv, cond
            else:
                return N_inv
        else:
            N_inv = np.diag( 1 / self.sig ** 2 )

            return N_inv
        

    def _get_radec(self):
        """Get the pulsar positions in RA and DEC."""
        psr_ra = self.psrs_phi
        psr_dec = (np.pi/2) - self.psrs_theta
        return psr_ra, psr_dec


    def _get_xi(self):
        """Calculate the angular separation between pulsar pairs.

        A function to compute the angular separation between pulsar pairs. This
        function will use a pair_idx array which is assigned upon construction 
        which ensures that the ordering of the pairs is consistent with the OS.

        Returns:
            np.ndarray: An array of pair separations.
        """
        psrs_ra, psrs_dec = self._get_radec()

        x = np.cos(psrs_ra)*np.cos(psrs_dec)
        y = np.sin(psrs_ra)*np.cos(psrs_dec)
        z = np.sin(psrs_dec)
        
        pos_vectors = np.array([x,y,z])

        a,b = self.pair_idx[:,0], self.pair_idx[:,1]

        xi = np.zeros( len(a) )
        # This effectively does a dot product of pulsar position vectors for all pairs a,b
        pos_dot = np.einsum('ij,ij->j', pos_vectors[:,a], pos_vectors[:, b])
        xi = np.arccos( pos_dot )

        return np.squeeze(xi)
    

    def _fplus_fcross(self, psrtheta, psrphi, gwtheta, gwphi):
        """Compute the antenna pattern functions. for each pulsar.

        This function comes primarily from NX01 (A propotype NANOGrav analysis 
        pipeline). This function supports vectorization for multiple pulsar positions 
        and multiple gw positions. 
        NOTE: This function uses the GW propogation direction for gwtheta and gwphi
        rather than the source direction (i.e. this method uses the vector from the
        source to the observer)
        
        Args:
            psrtheta (np.ndarray): An array of pulsar theta positions.
            psrphi (np.ndarray): An array of pulsar phi positions.
            gwtheta (np.ndarray): An array of GW theta propogation directions.
            gwphi (np.ndarray): An array of GW phi propogation directions.
        
        Returns:
            tuple: A tuple of two arrays, (Fplus, Fcross) containing the antenna 
                pattern functions for each pulsar.
        """
        # define variable for later use
        cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
        singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)

        # unit vectors to GW source
        m = np.array([singwphi, -cosgwphi, 0.0])
        n = np.array([-cosgwtheta*cosgwphi, -cosgwtheta*singwphi, singwtheta])
        omhat = np.array([-singwtheta*cosgwphi, -singwtheta*singwphi, -cosgwtheta])

        # pulsar location
        ptheta = psrtheta
        pphi = psrphi

        # use definition from Sesana et al 2010 and Ellis et al 2012
        phat = np.array([np.sin(ptheta)*np.cos(pphi), np.sin(ptheta)*np.sin(pphi),\
                np.cos(ptheta)])

        fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1 - np.dot(omhat, phat))
        fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 - np.dot(omhat, phat))

        return fplus, fcross
    

    def antenna_response(self):
        """A function to compute the antenna response matrix R_{ab,k}.

        This function computes the antenna response matrix R_{ab,k} where ab 
        represents the pulsar pair made of pulsars a and b, and k represents
        the pixel index. 
        NOTE: This function uses the GW propogation direction for gwtheta and gwphi
        rather than the source direction (i.e. this method uses the vector from the
        source to the observer)

        Returns:
            np.ndarray: An array of shape (npairs, npix) containing the antenna
                pattern response matrix.
        """
        npix = hp.nside2npix(self.nside)
        gwtheta,gwphi = hp.pix2ang(self.nside,np.arange(npix))

        FpFc = ac.signalResponse_fast(self.psrs_theta, self.psrs_phi, gwtheta, gwphi)
        Fp,Fc = FpFc[:,0::2], FpFc[:,1::2] 

        R_abk = np.zeros( (self.npairs,self.npix) )
        # Now lets do some multiplication
        for i,(a,b) in enumerate(self.pair_idx):
            R_abk[i] = Fp[a]*Fp[b] + Fc[a]*Fc[b]

        return R_abk
    

    def get_pure_HD(self):
        """Calculate the Hellings and Downs correlation for each pulsar pair.

        This function calculates the Hellings and Downs correlation for each pulsar
        pair. This is done by using the values of xi potentially supplied upon 
        construction. 

        Returns:
            np.ndarray: An array of HD correlation values for each pulsar pair.
        """
        #Return the theoretical HD curve given xi

        xx = (1 - np.cos(self.xi)) / 2.
        hd_curve = 1.5 * xx * np.log(xx) - xx / 4 + 0.5

        return hd_curve
    

    def orf_from_clm(self, params, include_scale=True):
        """A function to calculate the ORF from the clm values.

        This function calculates the ORF from the supplied clm values in params.
        If include_scale is True, it will include the isotropic scaling A^2 as params[0].
        Otherwise, it will assume that the isotropic scaling is 1.
        The rest of the params array are c_{lm} values from -m to m for each l.
        From there this function calculates the ORF for each pair given those clm values.

        Args:
            params (np.ndarray): An array of clm values.
            include_scale (bool): A flag to include the isotropic scaling A^2.

        Returns:
            np.ndarray: An array of ORF values for each pulsar pair.
        """
        # Using supplied clm values, calculate the corresponding power map
        # and calculate the ORF from that power map (convoluted, I know)

        if include_scale:
            amp2 = 10 ** params[0]
            clm = params[1:]
        else:
            amp2 = 1
            clm = params

        sh_map = ac.mapFromClm_fast(clm, nside = self.nside)

        orf = amp2 * np.dot(self.F_mat, sh_map)

        return np.longdouble(orf)
    

    def clmFromAlm(self, alm):
        """A function to convert alm values to clm values.

        Given an array of clm values, return an array of complex alm valuex
        NOTE: There is a bug in healpy for the negative m values. This function
        just takes the imaginary part of the abs(m) alm index.

        Args:
            alm (np.ndarray): An array of alm values.

        Returns:
            np.ndarray: An array of clm values.
        """
        #nalm = len(alm)
        #maxl = int(np.sqrt(9.0 - 4.0 * (2.0 - 2.0 * nalm)) * 0.5 - 1.5)  # Really?
        maxl = self.l_max
        nclm = (maxl + 1) ** 2

        # Check the solution. Went wrong one time..
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
    

    def fisher_matrix_sph(self, pair_cov=False):
        """A method to calculate the Fisher matrix for the spherical harmonic basis.

        Args:
            pair_cov (bool): A flag to use the pair covariance matrix if it was supplied

        Raises:
            ValueError: If pair_cov is True and no pair covariance matrix is supplied.

        Returns:
            np.array: The Fisher matrix for the spherical harmonic basis. [n_clm x n_clm]
        """
        F_mat_clm = self.Gamma_lm.T

        if pair_cov and self.pair_cov is None:
            raise ValueError("No pair covariance matrix supplied! Set it with set_data()")
        elif pair_cov:
            fisher_mat = F_mat_clm.T @ self.pair_cov_N_inv @ F_mat_clm
        else:
            fisher_mat = F_mat_clm.T @ self.pair_ind_N_inv @ F_mat_clm 

        return fisher_mat


    def fisher_matrix_pixel(self, pair_cov=False):
        """A method to calculate the Fisher matrix for the pixel basis.

        Args:
            pair_cov (bool): A flag to use the pair covariance matrix if it was supplied

        Raises:
            ValueError: If pair_cov is True and no pair covariance matrix is supplied.

        Returns:
            np.ndarray: The Fisher matrix for the pixel basis. [npix x npix]
        """
        if pair_cov and self.pair_cov is None:
            raise ValueError("No pair covariance matrix supplied! Set it with set_data()")
        elif pair_cov:
            fisher_mat = self.F_mat.T @ self.pair_cov_N_inv @ self.F_mat
        else:
            fisher_mat = self.F_mat.T @ self.pair_ind_N_inv @ self.F_mat
        return fisher_mat
    

    def max_lkl_pixel(self, cutoff = 0, return_fac1 = False, use_svd_reg = False, 
                      reg_type = 'l2', alpha = 0, pair_cov = False):
        """A method to calculate the maximum likelihood pixel values.

        This method calculates the maximum likelihood pixel values while allowing
        for covariance between pixels. This method is similar to that of the
        get_radiometer_map() method, but allows for covariance between pixels, and 
        can use regression to find solutions through forward modeling.

        Args:
            cutoff (float, optional): The minimum relative allowed singular value.
                    Only used if use_svd_reg is True. Defaults to 0.
            return_fac1 (bool): Whether to return the inverse of the Fisher matrix.
                    Defaults to False.
            use_svd_reg (bool): A flag to use linear solving with SVD regularization.
                    Defaults to False.
            reg_type (str): The type of regularization to use with astroML.linear_model.LinearRegression().
                    Defaults to 'l2'.
            alpha (int, optional): Optional jitter to add to the diagonal of the Fisher matrix
                    when using LinearRegression. Defaults to 0.
            pair_cov (bool): A flag to use the pair covariance matrix if it was supplied
                    at initialization or with set_data(). Defaults to False.
        
        Raises:
            ValueError: If pair_cov is True and no pair covariance matrix is supplied.

        Returns:
            tuple: A tuple of 4 np.ndarrays containing the pixel values, the pixel value errors,
                the condition number of the Fisher matrix, and the singular values of the Fisher matrix.
                If return_fac1 is True, it will also return the inverse of the Fisher matrix
                as an additional element of the tuple.
        """
        if pair_cov and self.pair_cov is None:
            raise ValueError("No pair covariance matrix supplied! Set it with set_data()")
        elif pair_cov:
            FNF = self.F_mat.T @ self.pair_cov_N_inv @ self.F_mat
        else:
            FNF = self.F_mat.T @ self.pair_ind_N_inv @ self.F_mat
        
        sv = sl.svd(FNF, compute_uv = False,)

        cn = np.max(sv) / np.min(sv)

        if use_svd_reg:
            abs_cutoff = cutoff * np.max(sv)
            fac1 = sl.pinvh( FNF, atol=abs_cutoff )
            
            if pair_cov:
                fac2 = self.F_mat.T @ self.pair_cov_N_inv @ self.rho
            else:
                fac2 = self.F_mat.T @ self.pair_ind_N_inv @ self.rho

            pow_err = np.sqrt(np.diag(fac1))
            power = fac1 @ fac2

        else:
            diag_identity = np.diag(np.full(self.F_mat.shape[1], 1))
            fac1r = sl.pinvh(FNF + alpha * diag_identity)
            pow_err = np.sqrt(np.diag(fac1r))
            clf = LinearRegression(regularization = reg_type, fit_intercept = False, 
                                   kwds = dict(alpha = alpha))
            
            if self.pair_cov is not None:
                clf.fit(self.F_mat, self.rho, self.pair_cov)
            else:
                clf.fit(self.F_mat, self.rho, self.sig)
            power = clf.coef_

        if return_fac1:
            return power, pow_err, cn, sv, fac1
        return power, pow_err, cn, sv
    

    def get_radiometer_map(self, pair_cov = True):
        """A method to get the radiometer pixel map.

        This method calculates the radiometer pixel map for all pixels. This method
        calculates power per pixel assuming there is no power in other pixels.
        NOTE: This function uses the GW propogation direction for each pixel
        rather than the source direction (i.e. this method uses the vector from the
        source to the observer). Use utils.invert_omega() to get the source direction
        instead!

        Args:
            pair_cov (bool): A flag to use the pair covariance matrix if it was supplied.

        Raises:
            ValueError: If pair_cov is True and no pair covariance matrix is supplied.

        Returns:
            tuple: A tuple of 2 np.ndarrays containing the pixel power map and the
                pixel power map error.
        """
        # Calculate dirty map
        if pair_cov and self.pair_cov is None:
            raise ValueError("No pair covariance matrix supplied! Set it with set_data()")
        elif pair_cov:
            dirty_map = self.F_mat.T @ self.pair_cov_N_inv @ self.rho
        else:
            dirty_map = self.F_mat.T @ self.pair_ind_N_inv @ self.rho
    
        # Calculate radiometer map, (i.e. no covariance between pixels)
        # If you take only the diagonal elements of the fisher matrix, 
        # you assume there is no covariance between pixels, so it will calculate
        # the power in each pixel assuming no power in others, i.e. radiometer!

        # Get the full fisher matrix
        fisher_mat = self.fisher_matrix_pixel()
        # Get only the diagonal elements and invert
        fisher_diag_inv = np.diag( 1/np.diag(fisher_mat) )
    
        # Solve
        radio_map = fisher_diag_inv @ dirty_map

        # The error needs to be normalized by the area of the pixel
        pix_area = hp.nside2pixarea(nside = self.nside)
        norm = 4 * np.pi / trapz(radio_map, dx = pix_area)

        radio_map_n = radio_map * norm
        radio_map_err = np.sqrt(np.diag(fisher_diag_inv)) * norm
    
        return radio_map_n, radio_map_err


    def max_lkl_clm(self, cutoff = None, use_svd_reg = False, reg_type = 'l2', alpha = 0,
                    pair_cov = False):
        """Compute the max likelihood clm values.

        A method to compute the maximum likelihood clm values. This method uses
        the Fisher matrix for the spherical harmonic basis to calculate the
        maximum likelihood clm values. This method allows for the use of SVD
        regularization or other regularization methods found in 
        astroML.linear_model.LinearRegression if use_svd_reg is False. If 
        use_svd_reg is True, it will solve the linear system using the SVD, otherwise
        it will use the LinearRegression to solve the system using forward modeling.

        Args:
            cutoff (float, optional): The minimum relative allowed singular value 
                for the SVD. Defaults to None.
            use_svd_reg (bool): A flag to use linear solving with SVD regularization.
                Defaults to False.
            reg_type (str, optional): The type of regularization to use with LinearRegression. 
                Defaults to 'l2'.
            alpha (float, optional): Optional jitter to add to the diagonal of the Fisher matrix
                when using LinearRegression. Defaults to 0.
            pair_cov (bool): A flag to use the pair covariance matrix if it was supplied.

        Returns:
            tuple: A tuple of 4 np.ndarrays containing the clm values, the clm value errors,
                the condition number of the Fisher matrix, and the singular values of the Fisher matrix.
        """
        F_mat_clm = self.Gamma_lm.T

        FNF_clm = self.fisher_matrix_sph(pair_cov)

        sv = sl.svd(FNF_clm, compute_uv=False)
        cn = np.max(sv) / np.min(sv)

        if use_svd_reg:
            # Swapped to np.linalg.pinv for easier implementation
            if cutoff is not None:
                fac1 = np.linalg.pinv(FNF_clm, rcond = cutoff)
            else:
                fac1 = np.linalg.pinv(FNF_clm)

            if pair_cov:
                fac2 = F_mat_clm.T @ self.pair_cov_N_inv @ self.rho
            else:
                fac2 = F_mat_clm.T @ self.pair_ind_N_inv @ self.rho

            clms = fac1 @ fac2
            clm_err = np.sqrt(np.diag(fac1))

        else:
            diag_identity = np.diag(np.ones(self.clm_size))

            fac1r = sl.pinvh(FNF_clm + alpha * diag_identity)

            clf = LinearRegression(regularization = reg_type, fit_intercept = False, kwds = dict(alpha = alpha))

            if pair_cov:
                clf.fit(F_mat_clm, self.rho, self.pair_cov)
            else:
                clf.fit(F_mat_clm, self.rho, self.sig)

            clms = clf.coef_

            clm_err = np.sqrt(np.diag(fac1r))

        return clms, clm_err, cn, sv


    def setup_lmfit_parameters(self):
        """A method to setup the lmfit Parameters object for the search.

        NOTE: This method is designed specifcally for the sqrt power basis. As
        such, it will not work for the power basis and may throw exceptions!

        Returns:
            lmfit.Parameters: The lmfit Parameters object
        """
        # TODO: Make this function work for other bases?
        # This shape is to fit with lmfit's Parameter.add_many();
        # Format is (name, value, vary, min, max, expr, brute_step)
        params = []

        if self.include_pta_monopole:
            # (name, value, vary, min, max, expr, brute_step)
            x = ['log10_A_mono', np.log10(nr.uniform(1e-2, 3)), True, np.log10(1e-2), np.log10(1e2), None, None]
            params.append(x)

            x = ['log10_A2', np.log10(nr.uniform(1e-2, 3)), True, np.log10(1e-2), np.log10(1e2), None, None]
            params.append(x)
        else:
            x = ['log10_A2', np.log10(nr.uniform(0, 3)), True, np.log10(1e-2), np.log10(1e2), None, None]
            params.append(x)

        # Now for non-monopole terms!
        for ll in range(self.blmax + 1):
            for mm in range(0, ll + 1):

                if ll == 0:
                    # (name, value, vary, min, max, expr, brute_step)
                    x = ['b_{}{}'.format(ll, mm), 1., False, None, None, None, None]
                    params.append(x)
                    
                elif mm == 0:
                    # (name, value, vary, min, max, expr, brute_step)
                    x = ['b_{}{}'.format(ll, mm), nr.uniform(-1, 1), True, None, None, None, None]
                    params.append(x)

                elif mm != 0:
                    # (name, value, vary, min, max, expr, brute_step)
                    #Amplitude is always >= 0; initial guess set to small non-zero value
                    x = ['b_{}{}_amp'.format(ll, mm), nr.uniform(0, 3), True, 0, None, None, None]
                    params.append(x)
                    
                    x = ['b_{}{}_phase'.format(ll, mm), nr.uniform(0, 2 * np.pi), True, 0, 2 * np.pi, None, None]
                    params.append(x)

        lmf_params = Parameters()
        lmf_params.add_many(*params)

        return lmf_params
    

    def max_lkl_sqrt_power(self, params = None, pair_cov = False, method = 'leastsq'):
        """A method to calculate the maximum likelihood b_lms for the sqrt power basis.

        This method uses lmfit to minimize the chi-square to find the maximum likelihood
        b_lms for the sqrt power basis. Post-processing help can be found in the utils 
        and lmfit documentation.

        Args:
            params (lmfit.Parameters, optional): The set of parameter to minimize. 
                    Defaults to an empty array.
            pair_cov (bool): A flag to use the pair covariance matrix if it was
                    supplied at initialization or with set_data(). Defaults to False
            method (str, optional): The method to use for minimization. This must be a valid
                    method for lmfit.Minimizer.minimize(). Defaults to 'leastsq'.

        Raises:
            ValueError: If pair_cov is True and no pair covariance matrix is supplied.
                    
        Returns:
            lmfit.Minimizer.minimize: The lmfit minimizer object for post-processing.
        """
        params = self.setup_lmfit_parameters() if params is None else params


        # Define L such that L @ L.T = C^-1
        if pair_cov: 
            # Use the Cholesky decomposition to get L
            if self._Lt_pc is None: # No reason to recompute these if done already
                self._Lt_pc = sl.cholesky(self.pair_cov_N_inv, lower = True).T
            Lt = self._Lt_pc
        else: 
            # Without pair covariance, L = L.T = diag(1/sig)
            if self._Lt_nopc is None: # No reason to recompute these if done already
                self._Lt_nopc = np.diag(1 / self.sig).T
            Lt = self._Lt_nopc


        def residuals(params):
            """A function to calculate the residuals for the lmfit minimizer.

            lmfit prefers residuals rather than scalars (more options and uncertainties 
            are returned more often). If we use pair covariance, our residuals are more 
            difficult than without. To combat this, we can define a whitening transformation 
            to remove covariances between residuals: 
            https://en.wikipedia.org/wiki/Whitening_transformation
        
            chi_square = -(r.T @ C^-1 @ r)
            = -(r.T @ L @ L.T @ r)
            = -(L.T @ r).T @ (L.T @ r)
            Therefore, our whitening transformation matrix is L.T
        
            This behavior is identical to what is done in scipy's curve_fit.

            Args:
                params (lmfit.Parameters): The set of parameters to minimize.
            
            Returns:
                np.ndarray: The noise-weighted (and whitened) residuals.
            """
            param_dict = params.valuesdict() # Get the input parameters (dict)            
            param_arr = np.array(list(param_dict.values())) # Convert to numpy array

            if self.include_pta_monopole:
                A_mono = 10**param_arr[0]
                A2 = 10**param_arr[1]
                clm = utils.convert_blm_params_to_clm(self, param_arr[2:])
            else:
                A_mono = 0
                A2 = 10**param_arr[0]
                clm = utils.convert_blm_params_to_clm(self, param_arr[1:])

            orf = np.sum(clm[:,None] * self.Gamma_lm, axis = 0)

            model_orf = A_mono + A2*orf
            
            r = self.rho - model_orf 

            return Lt @ r
        
        
        mini = lmfit.Minimizer(residuals, params)
        opt_params = mini.minimize(method)
        return opt_params


    def LogPrior(self, params):

        """A method to return the log-priors for the given set of parameters.

        This function works with 'power_basis' for now.

        Args:
            params (list or np.ndarray): An indexable object containing the parameters.

        Returns:
            float: The log-prior for the given parameters.
        """
        
        if self.mode == 'power_basis':
            
            log10_A2 = params[0]
            clm = params[1:] if type(params[1:]) is np.ndarray else np.array(params[1:])
        
            log10_A2_pdf = self.priors[0].get_logpdf(log10_A2)
            clm_pdf = np.sum([p.get_logpdf(c) for p,c in zip(self.priors[1:], clm)])

    
        return log10_A2_pdf + clm_pdf 


    def LogLikelihood(self, params):

        """A method to return the log-likelihood for the given set of parameters.

        This function works with 'power_basis' for now.

        Args:
            params (list or np.ndarray): An indexable object containing the parameters.

        Returns:
            float: The log-likelihood for the given parameters.
        """

        if self.mode == 'power_basis':
            
            A2 = 10 ** params[0]

            ### Fixing c_00 = root(4pi)
            clm_00 = np.sqrt(4*np.pi)
            clm_wo_00 = params[1:] if type(params[1:]) is np.ndarray else np.array(params[1:])
            clm = np.concatenate(([clm_00], clm_wo_00))
    
            sim_orf = A2 * (self.Gamma_lm.T @ clm[:, np.newaxis]) # (ncc x nclm) @ (nclm x 1) => RP - (ncc x 1)
            residual = self.rho[:, np.newaxis] - sim_orf # rho (ncc x 1) - RP (ncc x 1) => (ncc x 1)
        
            if self.pair_cov is not None:
                lik_num = (residual.T @ self.pair_cov_N_inv @ residual)[0][0] # (1 x ncc) @ (ncc x ncc) @ (ncc x 1) => (1 x 1)
                loglike = -0.5 * np.sum(lik_num + self._lik_denom)

            else:
                lik_num = (residual.ravel()**2) / (self.sig**2)
                lik_denom = np.longdouble(self.sig * np.sqrt(2 * np.pi))
                loglike = -0.5 * np.sum(lik_num + np.log(lik_denom))


        return loglike


    def set_ptmcmc(self, clm_prior_min=-5, clm_prior_max=5, log10_A2_prior_min=-2, log10_A2_prior_max=2, 
                   outdir='./ptmcmc', resume=False, save_anis_pta=False):

        """A method to return the PTMCMC sampler to perform sampling.

        This function works with 'power_basis' for now and raise ValueError if other modes are used.

        Args:
            clm_prior_min (float, optional): Lower bound for clm's uniform priors. Defaults to -5.
            clm_prior_max (float, optional): Lower bound for clm's uniform priors. Defaults to 5.
            log10_A2_prior_min (float, optional): Lower bound for log10_A2 uniform priors. Defaults to -2.
            log10_A2_prior_max (float, optional): Lower bound for log10_A2 uniform priors. Defaults to 2.
            outdir (str, optional): The path to save the chains. Defaults to './ptmcmc'
            resume (bool, optional): Whether to resume a previous run. Defaults to False
            save_anis_pta (bool, optional): Whether to save the anisotropy object for post prcessing help. Defaults to False.

        Returns:
            object: The PTMCMC sampler object.

        Raises:
            ValueError: If mode is not 'power_basis'.
        """

        if self.mode == 'power_basis':
            
            log10_A_prior = parameter.Uniform(log10_A2_prior_min, log10_A2_prior_max)("log10_A2")
            clm_prior = [parameter.Uniform(clm_prior_min, clm_prior_max)(f"c_{i}{j}") for i in range(1, self.l_max+1) for j in range(-i, i+1)]
            self.priors = [log10_A_prior, *clm_prior]
            self.param_names = [p.name for p in self.priors]
            self.ndim = len(self.param_names)

        else:
            raise ValueError("Bayesian Inference is only defined for the 'power_basis' for now!")
        
        # dimension of parameter space
        ndim = self.ndim

        # initial jump covariance matrix
        cov = np.diag(np.ones(ndim) * (0.1**2))

        # intialize sampler
        sampler = ptmcmc(ndim, self.LogLikelihood, self.LogPrior, cov, outDir=outdir, resume=resume)


        # save paramter list
        with open(os.path.join(outdir, "pars.txt"), "w") as f:
            for pn in self.param_names:
                f.write(pn + "\n")

        # save list of priors
        with open(os.path.join(outdir, "priors.txt"), "w") as f:
            for pr in self.priors:
                f.write(pr.__repr__() + "\n")

        # save the anisotropy object
        if save_anis_pta:
            saved_priors = self.priors
            del self.priors # cannot pickle with parameter class of enterprise.signals
            
            with open(os.path.join(outdir, "anis_pta.pickle"), "wb") as file:
                pickle.dump(self, file)

            self.priors = saved_priors # reassigning the priors
            

        return sampler



    def get_random_sample(self):
        """A method to get a random sample from the prior distribution.

        This method works with all modes, 'power_basis', 'sqrt_power_basis', and 'hybrid'.

        Returns:
            np.ndarray: A random sample for all parameters from the prior distribution.
        """

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
        """A method to calculate the amplitude scaling factor."""
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



class set_bilby(bilby.Likelihood):

    def __init__(self, anisotropy_pta, clm_prior_min=-5, clm_prior_max=5, log10_A2_prior_min=-2, log10_A2_prior_max=2, 
                 outdir='./bilby', save_anis_pta=False):

        """A class to perform bilby bayesian sampling with an anisotropy pta.

        This class can be used to perform bilby sampling by passing it the anisotropy pta.
        It constructs the anisotropic pta log-likelihood and log-prior functions internally and returns a
        bilby.Likelihood class object which can be simply supplied to bilby samplers to 
        perform bayesian inference with all the bilby-available samplers.
        
        Attributes:
            anisotropy_pta (object, required) : The anisotropic pta object defined by MAPS.
            clm_prior_min (float, optional): Lower bound for clm's uniform priors. Defaults to -5.
            clm_prior_max (float, optional): Lower bound for clm's uniform priors. Defaults to 5.
            log10_A2_prior_min (float, optional): Lower bound for log10_A2 uniform priors. Defaults to -2.
            log10_A2_prior_max (float, optional): Lower bound for log10_A2 uniform priors. Defaults to 2.
            save_anis_pta (bool, optional): Whether to save the anis_pta as a pickle.

        Returns:
            object: bilby.Likelihood class object.
    """


        self.anisotropy_pta = anisotropy_pta
        if self.anisotropy_pta.mode == 'power_basis':
            params = {"log10_A2": 0.0, **{f"c_{i}{j}": 0.0 for i in range(1, anisotropy_pta.l_max+1) for j in range(-i, i+1)}}
            self.clm_prior_min = clm_prior_min
            self.clm_prior_max = clm_prior_max
        else:
            raise ValueError("Bilby is currently only defined in 'power_basis'!")

        ### The sampler takes the parameter dictionary here to evaluate likelihood
        super().__init__(parameters=params)
        
        self.log10_A2_prior_min = log10_A2_prior_min
        self.log10_A2_prior_max = log10_A2_prior_max
        self.priors = self._priors()
        self.parameter_keys = list(self.priors.keys())
        self.ndim = len(self.parameter_keys)
        
        self.anisotropy_pta.param_names = self.parameter_keys
        self.anisotropy_pta.priors = self.priors
        self.anisotropy_pta.ndim = self.ndim
        

        # save the anisotropy object
        if save_anis_pta:
            os.makedirs(outdir, exist_ok=True) # creates the directory if doesn't exists
            with open(os.path.join(outdir, "anis_pta.pickle"), "wb") as file:
                pickle.dump(self.anisotropy_pta, file)
    

    def _priors(self):

        """A method to set the priors for the given mode.

        This function works with 'power_basis' for now.

        Returns:
            dict: Prior Dictionary.
        """

        if self.anisotropy_pta.mode == 'power_basis':
            priors = bilby.core.prior.PriorDict()
            priors["log10_A2"] = Uniform(name="log10_A2", minimum=self.log10_A2_prior_min, maximum=self.log10_A2_prior_max)

            for i in range(1, self.anisotropy_pta.l_max+1):
                for j in range(-i, i+1):
                    priors[f"c_{i}{j}"] = Uniform(name=f"c_{i}{j}", minimum=self.clm_prior_min, maximum=self.clm_prior_max)

        return priors


    def log_likelihood(self):

        """A method to return the log-likelihood for the set of parameters defined by the mode in anisotropy_pta.

        This function works with 'power_basis' for now.

        Returns:
            float: The log-likelihood for the given parameters.
        """

        if self.anisotropy_pta.mode == 'power_basis':
            
            params = np.array([self.parameters[key] for key in self.parameter_keys])
            A2 = 10 ** params[0]
            ### Fixing c_00 = root(4pi)
            clm_00 = np.sqrt(4*np.pi)
            clm_wo_00 = params[1:]
            clm = np.concatenate(([clm_00], clm_wo_00))
    
            sim_orf = A2 * (self.anisotropy_pta.Gamma_lm.T @ clm[:, np.newaxis]) # (ncc x nclm) @ (nclm x 1) => RP - (ncc x 1)
            residual = self.anisotropy_pta.rho[:, np.newaxis] - sim_orf # rho (ncc x 1) - RP (ncc x 1) => (ncc x 1)
        
            if self.anisotropy_pta.pair_cov is not None:
                lik_num = (residual.T @ self.anisotropy_pta.pair_cov_N_inv @ residual)[0][0] # (1 x ncc) @ (ncc x ncc) @ (ncc x 1) => (1 x 1)
                loglike = -0.5 * np.sum(lik_num + self.anisotropy_pta._lik_denom)

            else:
                lik_num = (residual.ravel()**2) / (self.anisotropy_pta.sig**2)
                lik_denom = np.longdouble(self.anisotropy_pta.sig * np.sqrt(2 * np.pi))
                loglike = -0.5 * np.sum(lik_num + np.log(lik_denom))


        return loglike


    
    #def noise_log_likelihood(self):

    #    """A method to return the noise (isotropic) log-likelihood for the given set of parameters defined by the mode in anisotropy_pta.

     #   This function works with 'power_basis' for now.

     #   Returns:
      #      float: The noise (isotropic) log-likelihood for the given parameters.
      #  """
        
       # if self.anisotropy_pta.mode == 'power_basis':
            
        #    A2 = 10 ** self.parameters['log10_A2']
            ### Fixing c_00 = root(4pi)
         #   clm_00 = np.sqrt(4*np.pi)
          #  clm_wo_00 = np.array([0.0 for i in range(1, self.anisotropy_pta.clm_size)])
           # clm = np.concatenate(([clm_00], clm_wo_00))
        
           # sim_iso_orf = A2 * (self.anisotropy_pta.Gamma_lm.T @ clm[:, np.newaxis]) ### RP - (ncc x 1)
            #residual = self.anisotropy_pta.rho[:, np.newaxis] - sim_iso_orf  ### (ncc x 1)

            #if self.anisotropy_pta.pair_cov is not None:
             #   lik_num = (residual.T @ self.anisotropy_pta.pair_cov_N_inv @ residual)[0][0] # (1 x ncc) @ (ncc x ncc) @ (ncc x 1) => (1 x 1)
              #  iso_loglike = -0.5 * np.sum(lik_num + self.anisotropy_pta._lik_denom)

            #else:
             #   lik_num = (residual.ravel()**2) / (self.anisotropy_pta.sig**2)
             #   lik_denom = np.longdouble(self.anisotropy_pta.sig * np.sqrt(2 * np.pi))
             #   iso_loglike = -0.5 * np.sum(lik_num + np.log(lik_denom))
        

        #return iso_loglike
    
