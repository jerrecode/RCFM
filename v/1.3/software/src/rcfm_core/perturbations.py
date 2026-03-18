#!/usr/bin/env python3
"""
RCFM Module - Perturbation Theory on S^3

This module implements perturbation theory for the RCFM model on the
3-sphere topology, including hyperspherical harmonics and mode calculations.

Author: MiniMax Agent
Version: 1.3
"""

import numpy as np
from typing import Tuple, Optional, Callable
from scipy import special, integrate
from .constants import PhysicalConstants, RCFMParameters
from .fluids import DualStreamFluid


class HypersphericalHarmonics:
    """
    Hyperspherical Harmonics on S^3

    Implements the eigenfunctions of the Laplace-Beltrami operator on S^3:
    ∇² Y_n^l_m = -(n² - 1) Y_n^l_m,  n ≥ 1

    where n is the principal quantum number and the degeneracy is n².
    """

    def __init__(self):
        """Initialize hyperspherical harmonics calculator."""
        pass

    def eigenvalue(self, n: int) -> float:
        """
        Calculate Laplacian eigenvalue for mode n.

        Args:
            n: Mode number (n ≥ 1)

        Returns:
            Eigenvalue -(n² - 1)
        """
        return -(n**2 - 1)

    def wavenumber(self, n: int, a: float) -> float:
        """
        Calculate wavenumber k_n = √(n² - 1) / a

        Args:
            n: Mode number
            a: Scale factor

        Returns:
            Wavenumber k_n
        """
        return np.sqrt(n**2 - 1) / a

    def degeneracy(self, n: int) -> int:
        """
        Calculate degeneracy of mode n.

        Args:
            n: Mode number

        Returns:
            Degeneracy n²
        """
        return n**2

    def create_mode_array(self, n_max: int, a: float) -> np.ndarray:
        """
        Create array of mode numbers and wavenumbers.

        Args:
            n_max: Maximum mode number
            a: Scale factor

        Returns:
            Array of shape (n_max, 3) with [n, k_n, degeneracy]
        """
        modes = []
        for n in range(1, n_max + 1):
            k_n = self.wavenumber(n, a)
            deg = self.degeneracy(n)
            modes.append([n, k_n, deg])

        return np.array(modes)


class ScalarPerturbations:
    """
    Scalar Perturbations on RCFM Background

    Implements the evolution of scalar perturbations (density contrasts)
    on the S^3 topology.
    """

    def __init__(self, params: RCFMParameters):
        """
        Initialize scalar perturbations solver.

        Args:
            params: RCFMParameters instance
        """
        self.params = params
        self.fluid = DualStreamFluid(params)
        self.harmonics = HypersphericalHarmonics()
        self.G = PhysicalConstants.G

    def jeffrey_equation(self, n: int, r: float, y: np.ndarray) -> np.ndarray:
        """
        Jeffrey's equation for scalar perturbations on S^3.

        This is the master equation for scalar mode evolution.

        Args:
            n: Mode number
            r: Radial coordinate
            y: State vector [δ_A, δ_B, v_A, v_B]

        Returns:
            Derivatives [dδ_A/dr, dδ_B/dr, dv_A/dr, dv_B/dr]
        """
        delta_A, delta_B, v_A, v_B = y

        # Get background quantities
        a = (r / self.params.Rmax)**2  # Scale factor
        H = self.fluid.hubble_parameter(r) if hasattr(self.fluid, 'hubble_parameter') else 1e-18

        # Wavenumber
        k_n = self.harmonics.wavenumber(n, a)

        # Equation of state
        w = self.fluid.equation_of_state(r)
        c_s² = w  # Sound speed squared

        # Growth rate
        ddelta_A_dr = -(1 + w) * v_A + self.params.Gamma_drag * delta_B
        ddelta_B_dr = v_B - self.params.Gamma_drag * delta_A

        # Velocity divergence
        dv_A_dr = (3 * c_s² - 1) * v_A - k_n**2 * delta_A / (a * H)
        dv_B_dr = -v_B

        return np.array([ddelta_A_dr, ddelta_B_dr, dv_A_dr, dv_B_dr])

    def solve_mode(self, n: int, r_start: float, r_end: float,
                   delta_A0: float = 1e-5,
                   delta_B0: float = 0.0,
                   v_A0: float = 0.0,
                   v_B0: float = 0.0) -> dict:
        """
        Solve scalar perturbation evolution for mode n.

        Args:
            n: Mode number
            r_start: Starting radial coordinate
            r_end: Ending radial coordinate
            delta_A0: Initial Phase A density contrast
            delta_B0: Initial Phase B density contrast
            v_A0: Initial Phase A velocity
            v_B0: Initial Phase B velocity

        Returns:
            Dictionary with solution arrays
        """
        from scipy.integrate import solve_ivp

        y0 = np.array([delta_A0, delta_B0, v_A0, v_B0])

        solution = solve_ivp(
            lambda r, y: self.jeffrey_equation(n, r, y),
            [r_start, r_end],
            y0,
            method='RK45',
            t_eval=np.linspace(r_start, r_end, 1000),
            rtol=1e-6,
            atol=1e-10
        )

        return {
            'r': solution.t,
            'delta_A': solution.y[0, :],
            'delta_B': solution.y[1, :],
            'v_A': solution.y[2, :],
            'v_B': solution.y[3, :],
            'n': n,
            'success': solution.success
        }

    def transfer_function(self, n: int, r: float) -> float:
        """
        Calculate transfer function T_n(r) for mode n.

        Args:
            n: Mode number
            r: Radial coordinate

        Returns:
            Transfer function value
        """
        # BBKS-like transfer function adapted for S^3
        a = (r / self.params.Rmax)**2
        k_n = self.harmonics.wavenumber(n, a)

        # Characteristic wavenumber
        k_eq = 0.05  # Matter-radiation equality wavenumber (h/Mpc)

        # Transfer function form
        if k_n < k_eq:
            T_n = 1.0
        else:
            T_n = (k_eq / k_n)**2

        return T_n


class TensorPerturbations:
    """
    Tensor Perturbations (Gravitational Waves) on S^3

    Implements the evolution of gravitational wave modes on the
    S^3 topology.
    """

    def __init__(self, params: RCFMParameters):
        """
        Initialize tensor perturbations solver.

        Args:
            params: RCFMParameters instance
        """
        self.params = params
        self.harmonics = HypersphericalHarmonics()
        self.G = PhysicalConstants.G

    def gw_equation(self, n: int, r: float, y: np.ndarray) -> np.ndarray:
        """
        Gravitational wave equation on S^3 background.

        h_n'' + 3H h_n' + [(n² - 1)/a² + m_GW²(r)] h_n = 0

        Args:
            n: Mode number (n ≥ 2 for tensor modes)
            r: Radial coordinate
            y: State vector [h, h']

        Returns:
            Derivatives [dh/dr, dh'/dr]
        """
        h, h_prime = y

        # Scale factor
        a = (r / self.params.Rmax)**2
        H = 1.0 / (2 * r)  # Approximate Hubble from a ∝ r²

        # Wavenumber term
        k_n² = (n**2 - 1) / a**2

        # Effective mass term from Phase B
        rho_A = self.params.rho_B0 * a**(-3)
        rho_B = self.params.rho_B0 * np.exp(-self.params.Gamma_drag * r)
        G_eff = self.G * (1 + self.params.beta * rho_B * (self.params.Rmax / a)**3)

        m_GW² = 8 * np.pi * G_eff * self.params.Gamma_drag * (rho_A - rho_B) / a**2

        # GW equation
        dh_dr = h_prime
        dh_prime_dr = -3 * H * h_prime - (k_n² + m_GW²) * h

        return np.array([dh_dr, dh_prime_dr])

    def solve_mode(self, n: int, r_start: float, r_end: float,
                   h0: float = 1e-10,
                   h_prime0: float = 0.0) -> dict:
        """
        Solve gravitational wave evolution for mode n.

        Args:
            n: Mode number (n ≥ 2)
            r_start: Starting radial coordinate
            r_end: Ending radial coordinate
            h0: Initial GW amplitude
            h_prime0: Initial GW derivative

        Returns:
            Dictionary with solution arrays
        """
        from scipy.integrate import solve_ivp

        y0 = np.array([h0, h_prime0])

        solution = solve_ivp(
            lambda r, y: self.gw_equation(n, r, y),
            [r_start, r_end],
            y0,
            method='RK45',
            t_eval=np.linspace(r_start, r_end, 1000),
            rtol=1e-8,
            atol=1e-12
        )

        return {
            'r': solution.t,
            'h': solution.y[0, :],
            'h_prime': solution.y[1, :],
            'n': n,
            'success': solution.success
        }

    def tensor_power_spectrum(self, n: int, r: float) -> float:
        """
        Calculate tensor power spectrum amplitude for mode n.

        P_T(n) = A_T * (k_n / k_0)^(n_T)

        Args:
            n: Mode number
            r: Radial coordinate

        Returns:
            Tensor power spectrum amplitude
        """
        a = (r / self.params.Rmax)**2
        k_n = self.harmonics.wavenumber(n, a)
        k_0 = 0.05  # Pivot scale (h/Mpc)

        # Tensor spectral index (from paper: n_T ≈ -(1 - n_s))
        n_T = -(1 - 0.965)  # Using Planck n_s

        # Amplitude (from constraint)
        A_T = 1e-10  # Placeholder

        return A_T * (k_n / k_0)**n_T


class PrimordialPowerSpectrum:
    """
    Primordial Power Spectrum from Singularity Boundary

    Implements the calculation of the primordial power spectrum
    from the quantum boundary condition at r = 0.
    """

    def __init__(self, params: RCFMParameters):
        """
        Initialize primordial spectrum calculator.

        Args:
            params: RCFMParameters instance
        """
        self.params = params
        self.harmonics = HypersphericalHarmonics()

    def scalar_spectrum(self, k: float) -> float:
        """
        Calculate scalar primordial power spectrum.

        P_S(k) = A_S * (k / k_0)^(n_S - 1)

        From the paper: n_S ≈ 0.965, generated by a ∝ r² boundary condition.

        Args:
            k: Wavenumber

        Returns:
            Power spectrum amplitude
        """
        # Pivot scale
        k_0 = 0.05  # Mpc/h

        # Spectral index (from a ∝ r² boundary condition)
        n_S = 0.965

        # Amplitude (from CMB normalization)
        A_S = 2.1e-9

        return A_S * (k / k_0)**(n_S - 1)

    def tensor_spectrum(self, k: float) -> float:
        """
        Calculate tensor primordial power spectrum.

        P_T(k) = A_T * (k / k_0)^(n_T)

        Args:
            k: Wavenumber

        Returns:
            Tensor power spectrum amplitude
        """
        k_0 = 0.05  # Mpc/h

        # Tensor spectral index
        n_T = -(1 - 0.965)  # Consistency relation

        # Amplitude (constrained by CMB)
        r_T = 0.06  # Tensor-to-scalar ratio (upper limit)
        A_T = r_T * 2.1e-9

        return A_T * (k / k_0)**n_T

    def generate_k_array(self, k_min: float = 1e-4,
                        k_max: float = 10.0,
                        n_points: int = 1000) -> np.ndarray:
        """
        Generate wavenumber array for spectrum calculations.

        Args:
            k_min: Minimum wavenumber
            k_max: Maximum wavenumber
            n_points: Number of points

        Returns:
            Wavenumber array (logarithmic spacing)
        """
        return np.logspace(np.log10(k_min), np.log10(k_max), n_points)


def compute_growth_factor(params: RCFMParameters,
                         z: float) -> float:
    """
    Compute matter growth factor D(z) for RCFM.

    From paper: D_+ ≈ a(1 + 3/5 Λ_RCFM)

    Args:
        params: RCFM parameters
        z: Redshift

    Returns:
        Growth factor
    """
    a = 1.0 / (1 + z)
    Lambda_RCFM = params.Lambda_RCFM(z)

    D = a * (1 + 3/5 * Lambda_RCFM)

    return D


def compute_growth_index(params: RCFMParameters,
                        z: float) -> float:
    """
    Compute growth index γ for RCFM.

    From paper: γ_RCFM = 0.55 + 0.05 ln(1 + Λ_RCFM)

    Args:
        params: RCFM parameters
        z: Redshift

    Returns:
        Growth index
    """
    Lambda_RCFM = params.Lambda_RCFM(z)

    gamma = 0.55 + 0.05 * np.log(1 + Lambda_RCFM)

    return gamma


if __name__ == "__main__":
    # Test the module
    print("RCFM Perturbation Theory Module Test")
    print("=" * 50)

    from constants import get_rcfm_defaults

    params = get_rcfm_defaults()

    # Test hyperspherical harmonics
    harmonics = HypersphericalHarmonics()
    print("\nHyperspherical Harmonics:")
    for n in [1, 2, 5, 10]:
        print(f"  n={n}: eigenvalue={harmonics.eigenvalue(n):.1f}, "
              f"k_n(a=1)={harmonics.wavenumber(n, 1.0):.3f}, "
              f"degeneracy={harmonics.degeneracy(n)}")

    # Test primordial spectrum
    spectrum = PrimordialPowerSpectrum(params)
    k_test = np.array([0.001, 0.01, 0.1, 1.0])
    print("\nPrimordial Power Spectrum:")
    for k in k_test:
        P_s = spectrum.scalar_spectrum(k)
        P_t = spectrum.tensor_spectrum(k)
        print(f"  k={k:.3f} Mpc/h: P_S={P_s:.2e}, P_T={P_t:.2e}")

    # Test growth factor
    print("\nGrowth Factor:")
    for z in [0, 0.5, 1.0, 2.0]:
        D = compute_growth_factor(params, z)
        gamma = compute_growth_index(params, z)
        print(f"  z={z}: D={D:.4f}, γ={gamma:.4f}")
