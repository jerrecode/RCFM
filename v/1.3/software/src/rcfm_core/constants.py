#!/usr/bin/env python3
"""
RCFM Core Module - Physical Constants and Model Parameters

This module defines the physical constants and model parameters
for the Radial-Cyclic Field Model (RCFM).

Author: MiniMax Agent
Version: 1.3
"""

import numpy as np
from typing import Dict, Tuple

# ============================================================================
# Physical Constants (SI units)
# ============================================================================

class PhysicalConstants:
    """Physical constants for cosmological calculations."""

    # Speed of light
    c = 299792458.0  # m/s

    # Gravitational constant
    G = 6.67430e-11  # m^3 kg^-1 s^-2

    # Planck constant
    h = 6.62607015e-34  # J s

    # Reduced Planck constant
    hbar = h / (2 * np.pi)  # J s

    # Planck mass
    Mpl = np.sqrt(hbar * c / (8 * np.pi * G))  # kg (reduced Planck mass)

    # Planck density
    rho_pl = c**6 / (hbar * G**3)  # kg/m^3
    rho_pl = rho_pl / (8 * np.pi)  # Reduced Planck density

    # Boltzmann constant
    k_B = 1.380649e-23  # J/K

    # Stefan-Boltzmann constant
    sigma_SB = (np.pi**2 * k_B**4) / (60 * hbar**3 * c**2)  # W m^-2 K^-4

    # Hubble constant (Planck 2018 value)
    H0_planck = 67.4  # km/s/Mpc
    H0_SI = H0_planck * 1000 / 3.086e22  # 1/s

    # Critical density today
    rho_crit_0 = 3 * H0_planck**2 / (8 * np.pi * G)  # kg/m^3 (Planck)


class RCFMParameters:
    """
    RCFM Model Parameters

    These are the free parameters of the RCFM model that need to be
    fitted to observational data.
    """

    def __init__(self,
                 beta: float = 1e-3,
                 rho_B0: float = 0.0,
                 Rmax: float = None,
                 Gamma_drag: float = None):
        """
        Initialize RCFM parameters.

        Args:
            beta: Drag coupling constant (dimensionless)
            rho_B0: Phase B density today (kg/m^3)
            Rmax: Critical radius (m)
            Gamma_drag: Drag kernel (m^3/kg/s)
        """
        self.beta = beta
        self.rho_B0 = rho_B0

        # Calculate Rmax from critical density if not provided
        if Rmax is None:
            self.Rmax = self._calculate_Rmax()
        else:
            self.Rmax = Rmax

        # Calculate Gamma_drag from matching condition if not provided
        if Gamma_drag is None:
            self.Gamma_drag = self._calculate_Gamma_drag()
        else:
            self.Gamma_drag = Gamma_drag

    def _calculate_Rmax(self) -> float:
        """Calculate Rmax from critical density condition."""
        # Rmax = c * t_age where t_age is the age of the universe
        # Using Planck 2018: t_age = 13.8 Gyr
        t_age = 13.8e9 * 365.25 * 24 * 3600  # seconds
        Rmax = PhysicalConstants.c * t_age
        return Rmax

    def _calculate_Gamma_drag(self) -> float:
        """Calculate Gamma_drag from matching condition."""
        # Gamma_drag ≈ 3H/rho_c at r=Rmax
        H_at_rmax = PhysicalConstants.H0_SI  # Approximation
        rho_c = PhysicalConstants.rho_crit_0
        Gamma_drag = 3 * H_at_rmax / rho_c
        return Gamma_drag

    def Lambda_RCFM(self, z: float) -> float:
        """
        Calculate Lambda_RCFM at redshift z.

        Args:
            z: Redshift

        Returns:
            Lambda_RCFM parameter
        """
        a = 1.0 / (1 + z)
        return 2 * self.beta * self.rho_B0 * (self.Rmax / a)**3

    def to_dict(self) -> Dict:
        """Convert parameters to dictionary."""
        return {
            'beta': self.beta,
            'rho_B0': self.rho_B0,
            'Rmax': self.Rmax,
            'Gamma_drag': self.Gamma_drag
        }


def get_planck_2018_params() -> Tuple[float, float, float, float]:
    """
    Get Planck 2018 cosmological parameters.

    Returns:
        Tuple of (H0, Omega_m, Omega_Lambda, Omega_k)
    """
    H0 = 67.4  # km/s/Mpc
    Omega_m = 0.315
    Omega_Lambda = 0.685
    Omega_k = 0.0

    return H0, Omega_m, Omega_Lambda, Omega_k


def get_rcfm_defaults() -> RCFMParameters:
    """
    Get default RCFM parameters consistent with Planck 2018.

    Returns:
        RCFMParameters instance
    """
    # Default parameters that reproduce LambdaCDM-like behavior
    beta = 1e-10  # Very small to approach LambdaCDM
    rho_B0 = PhysicalConstants.rho_crit_0 * 0.685  # Match Omega_Lambda

    return RCFMParameters(beta=beta, rho_B0=rho_B0)


if __name__ == "__main__":
    # Test the module
    print("RCFM Physical Constants and Parameters")
    print("=" * 50)

    params = get_rcfm_defaults()
    print(f"Default Beta: {params.beta:.2e}")
    print(f"Default rho_B0: {params.rho_B0:.2e} kg/m^3")
    print(f"Default Rmax: {params.Rmax:.2e} m")
    print(f"Default Gamma_drag: {params.Gamma_drag:.2e} m^3/kg/s")
    print()
    print(f"Lambda_RCFM(z=0): {params.Lambda_RCFM(0):.2e}")
    print(f"Lambda_RCFM(z=0.5): {params.Lambda_RCFM(0.5):.2e}")
    print(f"Lambda_RCFM(z=2.0): {params.Lambda_RCFM(2.0):.2e}")
