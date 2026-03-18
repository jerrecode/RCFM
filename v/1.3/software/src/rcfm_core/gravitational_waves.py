#!/usr/bin/env python3
"""
RCFM Module - Gravitational Waves

This module implements gravitational wave propagation and the stochastic
gravitational wave background for the RCFM model.

Author: MiniMax Agent
Version: 1.3
"""

import numpy as np
from typing import Tuple, Optional
from scipy import integrate, special
from scipy.interpolate import interp1d
from .constants import PhysicalConstants, RCFMParameters


class GravitationalWaves:
    """
    Gravitational Wave Calculator for RCFM

    Implements:
    - Tensor mode evolution on S^3
    - GW speed modifications
    - Stochastic background spectrum
    - GW170817 constraint
    """

    def __init__(self, params: RCFMParameters):
        """
        Initialize GW calculator.

        Args:
            params: RCFMParameters instance
        """
        self.params = params
        self.c = PhysicalConstants.c
        self.G = PhysicalConstants.G
        self.hbar = PhysicalConstants.hbar

    def gw_speed(self, z: float = 0) -> float:
        """
        Calculate GW propagation speed c_T.

        From paper: c_T² ≈ 1 - α_T(r)

        where α_T = d ln G_eff / d ln a

        Args:
            z: Redshift

        Returns:
            GW speed in units of c
        """
        Lambda_RCFM = self.params.Lambda_RCFM(z)

        # α_T from paper
        alpha_T = -6 * Lambda_RCFM / (1 + Lambda_RCFM)

        c_T_squared = 1 - alpha_T

        return np.sqrt(c_T_squared)

    def gw_mass_term(self, z: float = 0) -> float:
        """
        Calculate effective GW mass term m_GW².

        From paper: m_GW² = 8π G_eff Γ_drag (ρ_A - ρ_B) / a²

        Args:
            z: Redshift

        Returns:
            m_GW² in units of m^-2
        """
        a = 1.0 / (1 + z)

        # Effective G
        G_eff = self.G * (1 + self.params.Lambda_RCFM(z))

        # densities (simplified)
        rho_A = self.params.rho_B0 * a**(-3)
        rho_B = self.params.rho_B0 * np.exp(-self.params.Gamma_drag * self.params.Rmax * (1 - a))

        m_GW_squared = 8 * np.pi * G_eff * self.params.Gamma_drag * (rho_A - rho_B) / a**2

        return m_GW_squared

    def frequency_cutoff(self, z: float = 0) -> float:
        """
        Calculate low-frequency GW cutoff.

        From paper: f_cut = c * m_GW / (2π ħ)

        Args:
            z: Redshift

        Returns:
            Cutoff frequency in Hz
        """
        m_GW_squared = self.gw_mass_term(z)

        if m_GW_squared < 0:
            return 0

        m_GW = np.sqrt(m_GW_squared)

        # f = c * m / (2π ħ)
        f_cut = self.c * m_GW / (2 * np.pi * self.hbar)

        return f_cut

    def stochastic_background_primordial(self, f: np.ndarray) -> np.ndarray:
        """
        Calculate primordial stochastic GW background.

        From paper: Ω_GW^prim(f) has spectral index n_T ≈ -(1 - n_S)

        Args:
            f: Frequency array in Hz

        Returns:
            Ω_GW(f) array
        """
        # Pivot frequency
        f_0 = 0.05  # Hz (LIGO band)

        # Tensor spectral index
        n_T = -(1 - 0.965)  # From consistency relation

        # Amplitude (constrained)
        r_T = 0.06  # Upper limit on tensor-to-scalar ratio
        A_s = 2.1e-9  # Scalar amplitude

        A_T = r_T * A_s

        # Spectrum
        Omega_GW = np.zeros_like(f)

        # Only above cutoff
        f_cut = self.frequency_cutoff(0)
        mask = f > f_cut

        if np.any(mask):
            Omega_GW[mask] = A_T * (f[mask] / f_0)**n_T

        return Omega_GW

    def stochastic_background_phaseB(self, f: np.ndarray) -> np.ndarray:
        """
        Calculate Phase B source stochastic GW background.

        From paper: Ω_GW^(B)(f) is monochromatic at f^B ≈ 3.5×10^-19 Hz

        Args:
            f: Frequency array in Hz

        Returns:
            Ω_GW^B(f) array
        """
        # Characteristic frequency from Phase B
        f_B = 3.5e-19  # Hz (sub-astronomic band)

        # Amplitude (simplified)
        Lambda_RCFM = self.params.Lambda_RCFM(0)
        Omega_B = 1e-12 * Lambda_RCFM  # Very small

        # Monochromatic at f_B
        Omega_GW_B = np.zeros_like(f)
        mask = np.abs(f - f_B) < 0.1 * f_B
        Omega_GW_B[mask] = Omega_B

        return Omega_GW_B

    def stochastic_background_decoherence(self, f: np.ndarray) -> np.ndarray:
        """
        Calculate decoherence burst stochastic GW background.

        From paper: Burst at f_cycle from Phase A→B transition.

        Args:
            f: Frequency array in Hz

        Returns:
            Ω_GW^cycle(f) array
        """
        # Decoherence frequency
        # f_cycle = c / (2π λ_decoh)
        # λ_decoh ≈ ħ / sqrt(ρ_c)
        # For ρ_c = ρ_EW: f_cycle ≈ 10^-3 Hz

        f_cycle = 1e-3  # Hz (potentially LISA-observable)

        # Burst profile (simplified Gaussian)
        sigma_f = 0.1 * f_cycle  # Width

        # Amplitude (from transition energy)
        Lambda_RCFM = self.params.Lambda_RCFM(0)
        Omega_cycle = 1e-10 * Lambda_RCFM

        # Gaussian burst
        Omega_GW_cycle = Omega_cycle * np.exp(-((f - f_cycle)**2) / (2 * sigma_f**2))

        return Omega_GW_cycle

    def total_stochastic_background(self, f: np.ndarray) -> np.ndarray:
        """
        Calculate total stochastic GW background.

        From paper: Ω_GW = Ω_GW^prim + Ω_GW^B + Ω_GW^cycle

        Args:
            f: Frequency array in Hz

        Returns:
            Total Ω_GW(f)
        """
        Omega_prim = self.stochastic_background_primordial(f)
        Omega_B = self.stochastic_background_phaseB(f)
        Omega_cycle = self.stochastic_background_decoherence(f)

        return Omega_prim + Omega_B + Omega_cycle

    def gw170817_constraint(self) -> dict:
        """
        Check consistency with GW170817 constraint.

        From paper: |c_T - c|/c < 5×10^-16 at z ≈ 0.009

        Returns:
            Dictionary with constraint results
        """
        z_event = 0.009  # GW170817 redshift
        c_T = self.gw_speed(z_event)

        deviation = np.abs(c_T - 1.0)

        constraint_value = 5e-16

        passed = deviation < constraint_value

        return {
            'z': z_event,
            'c_T': c_T,
            'deviation': deviation,
            'constraint': constraint_value,
            'passed': passed,
            'headroom': constraint_value / deviation if passed else 0
        }

    def sensitivity_curve_LISA(self, f: np.ndarray) -> np.ndarray:
        """
        Calculate LISA sensitivity curve (approximate).

        Args:
            f: Frequency array in Hz

        Returns:
            Ω_effective(f) sensitivity
        """
        # Simplified LISA sensitivity
        # Based on arXiv:1702.00786

        f_knee = 0.01  # Hz
        f_min = 1e-5  # Hz
        f_max = 0.1  # Hz

        Omega_LISA = np.zeros_like(f)

        # Below knee frequency
        mask_low = (f > f_min) & (f < f_knee)
        Omega_LISA[mask_low] = 1e-2 * (f[mask_low] / f_knee)**(-0.5)

        # Above knee frequency
        mask_high = (f >= f_knee) & (f < f_max)
        Omega_LISA[mask_high] = 1e-2 * (f_knee / f[mask_high])**0.7

        return Omega_LISA

    def sensitivity_curve_PTA(self, f: np.ndarray) -> np.ndarray:
        """
        Calculate PTA (pulsar timing array) sensitivity curve.

        Args:
            f: Frequency array in Hz

        Returns:
            Ω_effective(f) sensitivity
        """
        # PTA sensitivity (nanohertz band)
        f_nano = 1e-9  # Hz
        f_nano_max = 1e-7  # Hz

        Omega_PTA = np.zeros_like(f)

        mask = (f > f_nano) & (f < f_nano_max)
        # Approximate ~f^-2/3 from brownian timing noise
        Omega_PTA[mask] = 1e-8 * (f[mask] / f_nano)**(-2.0/3.0)

        return Omega_PTA

    def sensitivity_curve_LIGO(self, f: np.ndarray) -> np.ndarray:
        """
        Calculate LIGO sensitivity curve (approximate).

        Args:
            f: Frequency array in Hz

        Returns:
            Ω_effective(f) sensitivity
        """
        # LIGO O3 sensitivity (simplified)
        f_low = 10  # Hz
        f_high = 1000  # Hz

        Omega_LIGO = np.zeros_like(f)

        mask = (f > f_low) & (f < f_high)
        # Approximate shape
        f_ref = 100  # Hz
        Omega_LIGO[mask] = 1e-9 * np.sqrt(f_ref / f[mask])

        return Omega_LIGO


def compute_gw_spectrum(params: RCFMParameters,
                       f_min: float = 1e-10,
                       f_max: float = 1e3,
                       n_points: int = 1000) -> dict:
    """
    Compute full GW spectrum and sensitivities.

    Args:
        params: RCFM parameters
        f_min: Minimum frequency in Hz
        f_max: Maximum frequency in Hz
        n_points: Number of frequency points

    Returns:
        Dictionary with spectrum and sensitivities
    """
    gw = GravitationalWaves(params)

    # Frequency array (logarithmic)
    f = np.logspace(np.log10(f_min), np.log10(f_max), n_points)

    # Compute spectra
    Omega_total = gw.total_stochastic_background(f)
    Omega_prim = gw.stochastic_background_primordial(f)
    Omega_B = gw.stochastic_background_phaseB(f)
    Omega_cycle = gw.stochastic_background_decoherence(f)

    # Sensitivities
    LISA = gw.sensitivity_curve_LISA(f)
    PTA = gw.sensitivity_curve_PTA(f)
    LIGO = gw.sensitivity_curve_LIGO(f)

    # GW170817 check
    constraint = gw.gw170817_constraint()

    return {
        'f': f,
        'Omega_total': Omega_total,
        'Omega_primordial': Omega_prim,
        'Omega_phaseB': Omega_B,
        'Omega_decoherence': Omega_cycle,
        'sensitivity_LISA': LISA,
        'sensitivity_PTA': PTA,
        'sensitivity_LIGO': LIGO,
        'gw170817': constraint
    }


if __name__ == "__main__":
    # Test the module
    print("RCFM Gravitational Waves Module Test")
    print("=" * 50)

    from constants import get_rcfm_defaults

    params = get_rcfm_defaults()

    # Initialize
    gw = GravitationalWaves(params)

    # GW speed
    c_T = gw.gw_speed(0)
    print(f"\nGW speed at z=0: c_T = {c_T:.15f} c")

    # GW170817 constraint
    constraint = gw.gw170817_constraint()
    print(f"\nGW170817 Constraint Check:")
    print(f"  Deviation: {constraint['deviation']:.2e}")
    print(f"  Constraint: {constraint['constraint']:.2e}")
    print(f"  Passed: {constraint['passed']}")
    print(f"  Headroom: {constraint['headroom']:.2e}x")

    # Frequency cutoff
    f_cut = gw.frequency_cutoff(0)
    print(f"\nFrequency cutoff: {f_cut:.2e} Hz")

    # Compute spectrum
    print("\nComputing GW spectrum...")
    result = compute_gw_spectrum(params, n_points=500)

    print(f"Frequency range: {result['f'].min():.2e} to {result['f'].max():.2e} Hz")
    print(f"Max Ω_GW: {result['Omega_total'].max():.2e}")

    print("\nTest completed successfully!")
