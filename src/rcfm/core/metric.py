#!/usr/bin/env python3
"""
RCFM Core Module - Metric Tensor and Curvature Calculations

This module implements the metric tensor and curvature calculations
for the Radial-Cyclic Field Model (RCFM).

Author: MiniMax Agent
Version: 1.3
"""

import numpy as np
from typing import Tuple, Optional
from scipy import integrate
from .constants import PhysicalConstants, RCFMParameters


class MetricRCFM:
    """
    RCFM Metric Tensor Class

    Implements the 4-dimensional hyperspherical metric:
    ds^2 = -A(r)*dr^2 + a(r)^2 * dOmega_3^2

    where dOmega_3^2 is the metric on unit S^3.
    """

    def __init__(self, params: RCFMParameters):
        """
        Initialize metric with RCFM parameters.

        Args:
            params: RCFMParameters instance
        """
        self.params = params
        self.c = PhysicalConstants.c

    def scale_factor(self, r: float) -> float:
        """
        Calculate scale factor a(r) from background solution.

        For the exact solution near singularity: a(r) ∝ r^2

        Args:
            r: Radial coordinate

        Returns:
            Scale factor a(r)
        """
        # Normalize to present-day scale factor = 1
        a0 = 1.0
        r0 = self.params.Rmax

        # Exact solution: a(r) ∝ r^2
        a = a0 * (r / r0)**2

        return a

    def lapse_function(self, r: float) -> float:
        """
        Calculate lapse function A(r).

        In synchronous gauge: A(r) = 1

        Args:
            r: Radial coordinate

        Returns:
            Lapse function A(r)
        """
        # Synchronous gauge
        return 1.0

    def metric_4D(self, r: float, chi: float, theta: float, phi: float) -> np.ndarray:
        """
        Calculate 4D metric tensor components.

        Args:
            r: Radial coordinate
            chi: S^3 polar angle
            theta: S^3 azimuthal angle
            phi: S^3 second azimuthal angle

        Returns:
            4x4 metric tensor g_mu_nu
        """
        A = self.lapse_function(r)
        a = self.scale_factor(r)

        # Metric components (mostly minus signature)
        g = np.zeros((4, 4))

        # g_rr = -A(r)
        g[0, 0] = -A

        # g_χχ = a(r)^2
        g[1, 1] = a**2

        # g_θθ = a(r)^2 * sin^2(chi)
        g[2, 2] = a**2 * np.sin(chi)**2

        # g_φφ = a(r)^2 * sin^2(chi) * sin^2(theta)
        g[3, 3] = a**2 * np.sin(chi)**2 * np.sin(theta)**2

        return g

    def inverse_metric(self, r: float, chi: float, theta: float, phi: float) -> np.ndarray:
        """
        Calculate inverse 4D metric tensor components.

        Args:
            r: Radial coordinate
            chi: S^3 polar angle
            theta: S^3 azimuthal angle
            phi: S^3 second azimuthal angle

        Returns:
            Inverse 4x4 metric tensor g^mu_nu
        """
        A = self.lapse_function(r)
        a = self.scale_factor(r)

        # Inverse metric components
        g_inv = np.zeros((4, 4))

        # g^rr = 1/A(r)
        g_inv[0, 0] = 1.0 / A

        # g^χχ = 1/a(r)^2
        g_inv[1, 1] = 1.0 / a**2

        # g^θθ = 1/(a(r)^2 * sin^2(chi))
        g_inv[2, 2] = 1.0 / (a**2 * np.sin(chi)**2)

        # g^φφ = 1/(a(r)^2 * sin^2(chi) * sin^2(theta))
        g_inv[3, 3] = 1.0 / (a**2 * np.sin(chi)**2 * np.sin(theta)**2)

        return g_inv

    def hubble_parameter(self, r: float) -> float:
        """
        Calculate generalized Hubble parameter H(r).

        H(r) = (1/a) * da/dtau = (a' / a) * 1/sqrt(A)

        Args:
            r: Radial coordinate

        Returns:
            Hubble parameter H(r)
        """
        a = self.scale_factor(r)
        A = self.lapse_function(r)

        # Derivative da/dr
        if r > 0:
            dr = r * 0.01  # Small step
            a2 = self.scale_factor(r + dr)
            da_dr = (a2 - a) / dr
        else:
            # Near singularity: a ∝ r^2
            da_dr = 2 * a / r if r > 0 else 0

        # H = a' / (a * sqrt(A))
        H = da_dr / (a * np.sqrt(A))

        return H

    def conformal_time(self, r: float) -> float:
        """
        Calculate conformal time eta = ∫ dr / (a * sqrt(A))

        Args:
            r: Radial coordinate

        Returns:
            Conformal time eta
        """
        def integrand(r_prime):
            a = self.scale_factor(r_prime)
            A = self.lapse_function(r_prime)
            return 1.0 / (a * np.sqrt(A))

        eta, error = integrate.quad(integrand, 0, r)
        return eta

    def S3_laplacian_eigenvalues(self, n: int) -> float:
        """
        Calculate eigenvalues of Laplacian on S^3.

        ∇^2 Y_n = -(n^2 - 1) Y_n,  n >= 1

        Args:
            n: Mode number

        Returns:
            Eigenvalue
        """
        return -(n**2 - 1)


class CurvatureCalculations:
    """
    Curvature calculations for RCFM metric.
    """

    def __init__(self, metric: MetricRCFM):
        """
        Initialize with metric.

        Args:
            metric: MetricRCFM instance
        """
        self.metric = metric

    def christoffel_symbols(self, r: float, chi: float, theta: float, phi: float) -> np.ndarray:
        """
        Calculate Christoffel symbols Γ^λ_μν.

        This is a placeholder - full implementation requires
        symbolic computation or numerical differentiation.

        Args:
            r: Radial coordinate
            chi: S^3 polar angle
            theta: S^3 azimuthal angle
            phi: S^3 second azimuthal angle

        Returns:
            Christoffel symbols (4x4x4 array)
        """
        # Simplified: return zeros (for testing)
        # Full implementation would compute derivatives of metric
        Gamma = np.zeros((4, 4, 4))
        return Gamma

    def ricci_tensor(self, r: float, chi: float, theta: float, phi: float) -> np.ndarray:
        """
        Calculate Ricci tensor R_μν.

        Args:
            r: Radial coordinate
            chi: S^3 polar angle
            theta: S^3 azimuthal angle
            phi: S^3 second azimuthal angle

        Returns:
            Ricci tensor (4x4 array)
        """
        # For FLRW-like metric with S^3 spatial sections:
        # R_00 = -3 a''/a
        # R_ij = (a''/a + 2 (a'/a)^2) g_ij

        a = self.metric.scale_factor(r)
        A = self.metric.lapse_function(r)

        # Approximate derivatives
        if r > 0:
            dr = r * 0.01
            a_prime = (self.metric.scale_factor(r + dr) - a) / dr
            a_prime2 = (self.metric.scale_factor(r + dr) - 2*a +
                       self.metric.scale_factor(r - dr)) / dr**2
        else:
            a_prime = 0
            a_prime2 = 0

        R = np.zeros((4, 4))

        # Time-time component
        R[0, 0] = -3 * a_prime2 / a

        # Spatial components (all equal for isotropic case)
        for i in range(1, 4):
            R[i, i] = (a_prime2 / a + 2 * (a_prime / a)**2) * a**2 * A

        return R

    def ricci_scalar(self, r: float) -> float:
        """
        Calculate Ricci scalar R.

        Args:
            r: Radial coordinate

        Returns:
            Ricci scalar
        """
        # For closed FLRW: R = 6 * (a''/a + (a'/a)^2 + 1/a^2)
        a = self.metric.scale_factor(r)
        A = self.metric.lapse_function(r)

        if r > 0:
            dr = r * 0.01
            a_prime = (self.metric.scale_factor(r + dr) - a) / dr
            a_prime2 = (self.metric.scale_factor(r + dr) - 2*a +
                       self.metric.scale_factor(r - dr)) / dr**2
        else:
            a_prime = 0
            a_prime2 = 0

        R = 6 * (a_prime2 / a + (a_prime / a)**2 + 1 / a**2)
        return R


def compute_background_evolution(params: RCFMParameters,
                                 r_array: np.ndarray) -> dict:
    """
    Compute background evolution for RCFM.

    Args:
        params: RCFM parameters
        r_array: Array of radial coordinates

    Returns:
        Dictionary with a(r), H(r), rho_A(r), rho_B(r)
    """
    metric = MetricRCFM(params)

    results = {
        'r': r_array,
        'a': np.zeros_like(r_array),
        'H': np.zeros_like(r_array),
        'rho_A': np.zeros_like(r_array),
        'rho_B': np.zeros_like(r_array)
    }

    for i, r in enumerate(r_array):
        results['a'][i] = metric.scale_factor(r)
        results['H'][i] = metric.hubble_parameter(r)

        # Density evolution (simplified)
        results['rho_A'][i] = params.rho_B0 * (results['a'][i]**(-3))
        results['rho_B'][i] = params.rho_B0 * np.exp(-params.Gamma_drag * r)

    return results


if __name__ == "__main__":
    # Test the module
    print("RCFM Metric and Curvature Calculations")
    print("=" * 50)

    from .constants import get_rcfm_defaults

    params = get_rcfm_defaults()
    metric = MetricRCFM(params)

    # Test at present day
    r_present = params.Rmax
    print(f"At present (r = {r_present:.2e} m):")
    print(f"  Scale factor: a = {metric.scale_factor(r_present):.4f}")
    print(f"  Lapse function: A = {metric.lapse_function(r_present):.4f}")
    print(f"  Hubble parameter: H = {metric.hubble_parameter(r_present):.2e} 1/s")

    # Test at early time
    r_early = r_present * 0.01
    print(f"\nAt early time (r = {r_early:.2e} m):")
    print(f"  Scale factor: a = {metric.scale_factor(r_early):.6f}")
    print(f"  Hubble parameter: H = {metric.hubble_parameter(r_early):.2e} 1/s")
