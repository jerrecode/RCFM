#!/usr/bin/env python3
"""
RCFM Module - Matter Power Spectrum

This module implements the calculation of the matter power spectrum
P(k) for the RCFM model, including the effects of the dual-stream
fluid and modified growth.

Author: MiniMax Agent
Version: 1.3
"""

import numpy as np
from typing import Tuple, Optional
from scipy import integrate, special
from scipy.interpolate import interp1d
from .constants import PhysicalConstants, RCFMParameters
from .perturbations import PrimordialPowerSpectrum, compute_growth_factor


class MatterPowerSpectrum:
    """
    Matter Power Spectrum Calculator for RCFM

    Computes P(k) including:
    - S^3 transfer function
    - Modified growth factor D(z)
    - Phase B drag modifications
    """

    def __init__(self, params: RCFMParameters):
        """
        Initialize matter power spectrum calculator.

        Args:
            params: RCFMParameters instance
        """
        self.params = params
        self.G = PhysicalConstants.G

    def transfer_function_BBKS(self, k: float) -> float:
        """
        Calculate BBKS transfer function on S^3.

        T(k) = ln(1 + 2.34q) / (2.34q) *
               [1 + 3.89q + (16.1q)² + (5.46q)³ + (6.71q)⁴]^-1/4

        where q = k / (Γ * Ω_m h² Mpc^-1)

        Args:
            k: Wavenumber in Mpc^-1

        Returns:
            Transfer function value
        """
        Omega_m = 0.315
        h = 0.674  # Hubble / 100

        # Shape parameter (adapted for closed universe)
        Gamma = Omega_m * h * (1 - 0.5 * self.params.Lambda_RCFM(0))

        q = k / (Gamma * Omega_m * h**2)

        # BBKS form
        T_k = np.log(1 + 2.34 * q) / (2.34 * q)
        T_k /= (1 + 3.89 * q + (16.1 * q)**2 + (5.46 * q)**3 + (6.71 * q)**4)**0.25

        return T_k

    def s3_transfer_function(self, k: float, z: float = 0) -> float:
        """
        Calculate S^3 transfer function correction.

        From paper: T_S³(k) introduces discrete mode corrections
        for wavenumbers near k_n = √(n²-1) / a(z)

        Args:
            k: Wavenumber in Mpc^-1
            z: Redshift

        Returns:
            S^3 correction factor
        """
        a = 1.0 / (1 + z)
        Lambda_RCFM = self.params.Lambda_RCFM(z)

        # Mode spacing
        delta_k = 1.0 / a  # Approximate spacing between S^3 modes

        # Resonances at mode positions
        T_S3 = 1.0
        for n in range(1, 100):
            k_n = np.sqrt(n**2 - 1) / a
            width = 0.1 * k_n  # Resonance width

            # Lorentzian enhancement/suppression
            T_S3 += 0.01 * Lambda_RCFM * k_n**2 / ((k - k_n)**2 + width**2)

        return T_S3

    def growth_factor_modified(self, z: float) -> float:
        """
        Calculate modified growth factor D(z) for RCFM.

        From paper: D_+ ≈ a(1 + 3/5 Λ_RCFM)

        Args:
            z: Redshift

        Returns:
            Growth factor
        """
        return compute_growth_factor(self.params, z)

    def compute_Pk(self, k: np.ndarray, z: float = 0) -> np.ndarray:
        """
        Compute matter power spectrum P(k) at redshift z.

        P(k, z) = A_s * k^(n_s-1) * T(k)² * D(z)²

        Args:
            k: Wavenumber array in Mpc^-1
            z: Redshift

        Returns:
            Power spectrum array
        """
        # Get primordial spectrum
        spectrum = PrimordialPowerSpectrum(self.params)

        # Compute P(k) at z=0
        Pk_z0 = np.zeros_like(k, dtype=float)

        for i, k_val in enumerate(k):
            # Skip k=0
            if k_val <= 0:
                Pk_z0[i] = 0
                continue

            # Primordial spectrum
            P_s = spectrum.scalar_spectrum(k_val)

            # Transfer function
            T_k = self.transfer_function_BBKS(k_val)

            # S^3 correction
            T_S3 = self.s3_transfer_function(k_val, z=0)

            # Combine
            Pk_z0[i] = P_s * T_k**2 * T_S3

        # Redshift evolution
        D_z = self.growth_factor_modified(z)
        D_0 = self.growth_factor_modified(0)

        Pk_z = Pk_z0 * (D_z / D_0)**2

        return Pk_z

    def compute_Pk_rcfm_vs_lcdm(self, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute P(k) for both RCFM and ΛCDM for comparison.

        Args:
            k: Wavenumber array in Mpc^-1

        Returns:
            Tuple of (P_RCFM, P_LCDM)
        """
        P_RCFM = self.compute_Pk(k, z=0)

        # ΛCDM (no Phase B corrections)
        params_lcdm = self.params
        params_lcdm.beta = 0  # Disable RCFM corrections
        params_lcdm.rho_B0 = 0

        # Recompute with ΛCDM parameters
        from .perturbations import PrimordialPowerSpectrum
        spectrum = PrimordialPowerSpectrum(params_lcdm)

        P_LCDM = np.zeros_like(k, dtype=float)
        for i, k_val in enumerate(k):
            if k_val <= 0:
                P_LCDM[i] = 0
                continue

            P_s = spectrum.scalar_spectrum(k_val)
            T_k = self.transfer_function_BBKS(k_val)
            P_LCDM[i] = P_s * T_k**2

        return P_RCFM, P_LCDM

    def sigma8_RCFM(self, z: float = 0) -> float:
        """
        Calculate σ₈ for RCFM.

        σ₈² = ∫ P(k) * W(kR)² * k² dk / (2π²)

        where W(kR) is a top-hat window function with R = 8 h⁻¹ Mpc

        Args:
            z: Redshift

        Returns:
            σ₈ value
        """
        # Wavenumber array
        k = np.logspace(-3, 1, 1000)  # Mpc^-1

        # P(k)
        Pk = self.compute_Pk(k, z)

        # Window function
        R = 8.0  # Mpc/h
        R_Mpc = R / 0.674  # Convert to Mpc

        # Top-hat in Fourier space
        W = 3 * (np.sin(k * R_Mpc) - k * R_Mpc * np.cos(k * R_Mpc)) / (k * R_Mpc)**3

        # Variance
        integrand = Pk * W**2 * k**2 / (2 * np.pi**2)
        sigma8_sq, _ = integrate.quad(lambda k_val:
            np.interp(k_val, k, integrand) if k_val > k[0] and k_val < k[-1] else 0,
            k[0], k[-1])

        return np.sqrt(sigma8_sq)

    def fsigma8_RCFM(self, z: float) -> float:
        """
        Calculate fσ₈ for RCFM.

        fσ₈ = f(z) * σ₈(z)

        where f(z) = d ln D / d ln a ≈ Ω_m(z)^γ

        Args:
            z: Redshift

        Returns:
            fσ₈ value
        """
        from .perturbations import compute_growth_index

        # Growth rate f = d ln D / d ln a
        gamma = compute_growth_index(self.params, z)
        Omega_m_z = 0.315 * (1 + z)**3 / self.params.Lambda_RCFM(z)  # Approximate

        f = Omega_m_z**gamma

        # σ₈ at this redshift
        sigma8_z = self.sigma8_RCFM(z)

        return f * sigma8_z


def compute_matter_power_spectrum(params: RCFMParameters,
                                k_min: float = 1e-4,
                                k_max: float = 10.0,
                                n_points: int = 500,
                                z: float = 0) -> dict:
    """
    Convenience function to compute matter power spectrum.

    Args:
        params: RCFM parameters
        k_min: Minimum wavenumber in Mpc^-1
        k_max: Maximum wavenumber in Mpc^-1
        n_points: Number of points
        z: Redshift

    Returns:
        Dictionary with k and P(k)
    """
    calculator = MatterPowerSpectrum(params)

    k = np.logspace(np.log10(k_min), np.log10(k_max), n_points)
    Pk = calculator.compute_Pk(k, z)

    return {
        'k': k,
        'Pk': Pk,
        'z': z,
        'units': 'Mpc³/h³'
    }


if __name__ == "__main__":
    # Test the module
    print("RCFM Matter Power Spectrum Module Test")
    print("=" * 50)

    from constants import get_rcfm_defaults

    params = get_rcfm_defaults()

    # Initialize calculator
    mps = MatterPowerSpectrum(params)

    # Compute spectrum
    k = np.logspace(-4, 1, 100)
    Pk = mps.compute_Pk(k, z=0)

    print(f"\nComputed P(k) at {len(k)} wavenumbers")
    print(f"k range: {k.min():.2e} to {k.max():.2e} Mpc^-1")
    print(f"P(k) range: {Pk.min():.2e} to {Pk.max():.2e}")

    # Compare with ΛCDM
    P_RCFM, P_LCDM = mps.compute_Pk_rcfm_vs_lcdm(k)
    print(f"\nRCFM/ΛCDM ratio at k=0.1: {P_RCFM[k>0.05][0] / P_LCDM[k>0.05][0]:.4f}")

    # σ₈
    sigma8 = mps.sigma8_RCFM(z=0)
    print(f"\nσ₈ at z=0: {sigma8:.4f}")

    # fσ₈
    for z in [0, 0.5, 1.0]:
        fsigma8 = mps.fsigma8_RCFM(z)
        print(f"fσ₈ at z={z}: {fsigma8:.4f}")

    print("\nTest completed successfully!")
