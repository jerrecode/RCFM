#!/usr/bin/env python3
"""
RCFM Module - Likelihood Analysis

This module implements likelihood analysis for parameter estimation
and model comparison for the RCFM model.

Author: MiniMax Agent
Version: 1.3
"""

import numpy as np
from typing import Tuple, Optional, Callable
from scipy import stats, optimize
from scipy.interpolate import interp1d
from .constants import PhysicalConstants, RCFMParameters
from .cmb_spectrum import CMBAngularSpectrum
from .matter_power import MatterPowerSpectrum


class RCFMUniformPrior:
    """
    Uniform prior distribution for RCFM parameters.
    """

    def __init__(self, bounds: dict):
        """
        Initialize uniform prior.

        Args:
            bounds: Dictionary with parameter bounds
        """
        self.bounds = bounds

    def log_prior(self, params: dict) -> float:
        """
        Calculate log prior probability.

        Args:
            params: Parameter dictionary

        Returns:
            Log prior probability
        """
        for key, value in params.items():
            if key not in self.bounds:
                return -np.inf

            low, high = self.bounds[key]
            if value < low or value > high:
                return -np.inf

        return 0.0  # Uniform prior


class RCFMLikelihood:
    """
    Joint Likelihood for RCFM Parameter Estimation

    Combines:
    - CMB likelihood (Planck 2018)
    - BAO likelihood (DESI 2024)
    - BBN likelihood (Cooke et al. 2018)
    - RSD likelihood (BOSS/eBOSS)
    - GW170817 constraint
    """

    def __init__(self, data_dir: str = None):
        """
        Initialize likelihood calculator.

        Args:
            data_dir: Directory containing observational data
        """
        self.data_dir = data_dir
        self.cmb = None  # Initialized when needed
        self.mps = None

    def log_likelihood_CMB(self, params: RCFMParameters) -> float:
        """
        Calculate CMB likelihood contribution.

        Args:
            params: RCFM parameters

        Returns:
            Log likelihood
        """
        if self.cmb is None:
            self.cmb = CMBAngularSpectrum(params)

        # Compute spectrum
        Cl_rcfm = self.cmb.compute_Cl(l_max=2500)

        # Load Planck data (placeholder)
        Cl_planck = self.cmb._load_planck_data(2500)

        # χ² calculation (simplified)
        l = Cl_rcfm['l']
        Cl_diff = Cl_rcfm['Cl'] - Cl_planck

        # Covariance (simplified diagonal)
        Cov = np.diag(Cl_rcfm['Cl']**2)

        chi2 = np.sum(Cl_diff**2 / Cov)

        return -0.5 * chi2

    def log_likelihood_BAO(self, params: RCFMParameters) -> float:
        """
        Calculate BAO likelihood contribution.

        DESI 2024 BAO measurements.

        Args:
            params: RCFM parameters

        Returns:
            Log likelihood
        """
        # DESI BAO observable: α (distance ratio)
        # α = D_A(z) / D_A^fid(z) or H(z) / H^fid(z)

        # Simplified: just use Λ_RCFM at DESI redshifts
        z_desi = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.1])

        chi2 = 0
        for z in z_desi:
            Lambda = params.Lambda_RCFM(z)
            # Constraint: Λ < 1 (for perturbative validity)
            if Lambda > 1:
                return -np.inf

            chi2 += Lambda**2 / (0.01)**2  # Approximate constraint

        return -0.5 * chi2

    def log_likelihood_BBN(self, params: RCFMParameters) -> float:
        """
        Calculate BBN likelihood contribution.

        Cooke et al. 2018 deuterium constraint.

        Args:
            params: RCFM parameters

        Returns:
            Log likelihood
        """
        # BBN constrains Omega_b h²
        # And indirectly constrains Λ_RCFM at z ~ 6×10^8

        Lambda_BBN = params.Lambda_RCFM(6e8)

        # Constraint from BBN (simplified)
        chi2 = (Lambda_BBN / 1e-3)**2

        return -0.5 * chi2

    def log_likelihood_RSD(self, params: RCFMParameters) -> float:
        """
        Calculate RSD (Redshift Space Distortions) likelihood.

        BOSS/eBOSS fσ₈ measurements.

        Args:
            params: RCFM parameters

        Returns:
            Log likelihood
        """
        if self.mps is None:
            self.mps = MatterPowerSpectrum(params)

        # fσ₈ measurements (approximate)
        z_rsd = np.array([0.38, 0.51, 0.61])
        fsigma8_obs = np.array([0.44, 0.46, 0.48])
        fsigma8_err = np.array([0.05, 0.04, 0.04])

        chi2 = 0
        for i, z in enumerate(z_rsd):
            fsigma8_theory = self.mps.fsigma8_RCFM(z)
            chi2 += ((fsigma8_theory - fsigma8_obs[i]) / fsigma8_err[i])**2

        return -0.5 * chi2

    def log_likelihood_GW170817(self, params: RCFMParameters) -> float:
        """
        Calculate GW170817 constraint contribution.

        Args:
            params: RCFM parameters

        Returns:
            Log likelihood (Gaussian constraint)
        """
        from .gravitational_waves import GravitationalWaves

        gw = GravitationalWaves(params)
        c_T = gw.gw_speed(z=0.009)

        # Constraint: |c_T - 1| < 5×10^-16
        deviation = np.abs(c_T - 1.0)
        constraint = 5e-16

        # Very tight Gaussian constraint
        sigma = constraint / 3  # 3σ equivalent

        chi2 = (deviation / sigma)**2

        return -0.5 * chi2

    def log_likelihood(self, params: RCFMParameters) -> float:
        """
        Calculate total log likelihood.

        From paper: χ² = χ²_CMB + χ²_BAO + χ²_BBN + χ²_RSD + χ²_GW170817

        Args:
            params: RCFM parameters

        Returns:
            Total log likelihood
        """
        # Check parameter validity
        if params.Lambda_RCFM(0) > 1:
            return -np.inf

        L_CMB = self.log_likelihood_CMB(params)
        L_BAO = self.log_likelihood_BAO(params)
        L_BBN = self.log_likelihood_BBN(params)
        L_RSD = self.log_likelihood_RSD(params)
        L_GW = self.log_likelihood_GW170817(params)

        return L_CMB + L_BAO + L_BBN + L_RSD + L_GW


class MCMCSampler:
    """
    Simple MCMC sampler for RCFM parameters.

    This is a placeholder - use emcee or PyMC3 for production.
    """

    def __init__(self, likelihood, prior, nwalkers: int = 32):
        """
        Initialize MCMC sampler.

        Args:
            likelihood: RCFMLikelihood instance
            prior: Prior distribution
            nwalkers: Number of walkers
        """
        self.likelihood = likelihood
        self.prior = prior
        self.nwalkers = nwalkers

    def log_probability(self, params_vec: np.ndarray) -> float:
        """
        Calculate log probability (prior + likelihood).

        Args:
            params_vec: Parameter vector [beta, rho_B0]

        Returns:
            Log probability
        """
        # Convert to dict
        params_dict = {
            'beta': params_vec[0],
            'rho_B0': params_vec[1]
        }

        # Prior
        log_prior = self.prior.log_prior(params_dict)
        if not np.isfinite(log_prior):
            return -np.inf

        # Create params object
        rcfm_params = RCFMParameters(**params_dict)

        # Likelihood
        log_like = self.likelihood.log_likelihood(rcfm_params)

        return log_prior + log_like

    def run_simple(self, nsteps: int = 1000,
                  burnin: int = 100) -> dict:
        """
        Run simple Metropolis-Hastings MCMC.

        Args:
            nsteps: Number of steps
            burnin: Burn-in steps

        Returns:
            Chain dictionary
        """
        # Initialize at prior mean
        x = np.array([1e-10, 0.5])

        # Storage
        chain = np.zeros((nsteps, 2))
        log_prob = np.zeros(nsteps)

        # Acceptance counter
        accepted = 0

        for i in range(nsteps):
            # Propose step
            x_proposed = x + 0.1 * np.random.randn(2)
            x_proposed = np.abs(x_proposed)  # Keep positive

            # Calculate acceptance ratio
            log_prob_new = self.log_probability(x_proposed)
            log_prob_old = self.log_probability(x)

            alpha = np.exp(log_prob_new - log_prob_old)

            # Accept/reject
            if np.random.rand() < alpha:
                x = x_proposed
                accepted += 1

            chain[i] = x
            log_prob[i] = log_prob_new

        # Discard burn-in
        chain = chain[burnin:]
        log_prob = log_prob[burnin:]

        return {
            'chain': chain,
            'log_prob': log_prob,
            'acceptance_fraction': accepted / nsteps
        }


def compute_AIC(params: RCFMParameters, likelihood: RCFMLikelihood) -> float:
    """
    Compute Akaike Information Criterion.

    AIC = 2k - 2 ln(L)

    Args:
        params: Best-fit parameters
        likelihood: Likelihood instance

    Returns:
        AIC value
    """
    k = 2  # Number of parameters
    log_L = likelihood.log_likelihood(params)

    return 2 * k - 2 * log_L


def compute_BIC(params: RCFMParameters, likelihood: RCFMLikelihood,
                n_data: int) -> float:
    """
    Compute Bayesian Information Criterion.

    BIC = k ln(n) - 2 ln(L)

    Args:
        params: Best-fit parameters
        likelihood: Likelihood instance
        n_data: Number of data points

    Returns:
        BIC value
    """
    k = 2  # Number of parameters
    log_L = likelihood.log_likelihood(params)

    return k * np.log(n_data) - 2 * log_L


if __name__ == "__main__":
    # Test the module
    print("RCFM Likelihood Analysis Module Test")
    print("=" * 50)

    from constants import get_rcfm_defaults

    params = get_rcfm_defaults()

    # Initialize likelihood
    like = RCFMLikelihood()

    # Test individual likelihoods
    print("\nTesting individual likelihoods:")
    L_CMB = like.log_likelihood_CMB(params)
    print(f"  CMB: {L_CMB:.2f}")

    L_BAO = like.log_likelihood_BAO(params)
    print(f"  BAO: {L_BAO:.2f}")

    L_BBN = like.log_likelihood_BBN(params)
    print(f"  BBN: {L_BBN:.2f}")

    L_RSD = like.log_likelihood_RSD(params)
    print(f"  RSD: {L_RSD:.2f}")

    L_GW = like.log_likelihood_GW170817(params)
    print(f"  GW170817: {L_GW:.2f}")

    L_total = like.log_likelihood(params)
    print(f"  Total: {L_total:.2f}")

    # Test MCMC
    print("\nRunning simple MCMC (10 steps)...")
    prior = RCFMUniformPrior({
        'beta': [1e-20, 1e-2],
        'rho_B0': [1e-30, 1e-20]
    })

    sampler = MCMCSampler(like, prior, nwalkers=4)
    result = sampler.run_simple(nsteps=10, burnin=2)

    print(f"Acceptance fraction: {result['acceptance_fraction']:.2f}")

    print("\nTest completed successfully!")
