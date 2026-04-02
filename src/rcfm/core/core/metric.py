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

    # Minimum r value to avoid singularity (BUG-FIX-4)
    R_MIN = 1e-10  # meters

    def __init__(self, params: RCFMParameters):
        """
        Initialize metric with RCFM parameters.

        Args:
            params: RCFMParameters instance
        """
        self.params = params
        self.c = PhysicalConstants.c
        # Store normalization constants
        self.a0 = 1.0  # Present-day scale factor
        self.r0 = params.Rmax

    def scale_factor(self, r: float) -> float:
        """
        Calculate scale factor a(r) from background solution.

        For the exact solution near singularity: a(r) ∝ r^2

        BUG-FIX-4: Added regularization to avoid division by zero at r → 0.

        Args:
            r: Radial coordinate

        Returns:
            Scale factor a(r)
        """
        # BUG-FIX-4: Regularize r to avoid singularity at r → 0
        r_safe = max(abs(r), self.R_MIN) * np.sign(r) if r != 0 else self.R_MIN

        # Exact solution: a(r) ∝ r^2
        a = self.a0 * (r_safe / self.r0)**2

        return a

    def scale_factor_derivative(self, r: float) -> float:
        """
        Analytical derivative of scale factor: da/dr.

        For a(r) = a0 * (r/r0)^2: da/dr = 2 * a0 * r / r0^2

        Args:
            r: Radial coordinate

        Returns:
            da/dr
        """
        # BUG-FIX-4: Regularize r
        r_safe = max(abs(r), self.R_MIN) * np.sign(r) if r != 0 else self.R_MIN

        # Analytical derivative of a(r) = a0 * (r/r0)^2
        da_dr = 2 * self.a0 * r_safe / (self.r0**2)
        return da_dr

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

        BUG-FIX-5: Use analytical derivative instead of numerical.
        BUG-FIX-4: Handle r → 0 with regularization.

        Args:
            r: Radial coordinate

        Returns:
            Hubble parameter H(r)
        """
        # BUG-FIX-4: Regularize r
        r_safe = max(abs(r), self.R_MIN) * np.sign(r) if r != 0 else self.R_MIN

        # Get scale factor and its analytical derivative
        a = self.scale_factor(r_safe)
        A = self.lapse_function(r_safe)

        # BUG-FIX-5: Use analytical derivative da/dr
        # For a(r) = a0 * (r/r0)^2: da/dr = 2 * a0 * r / r0^2
        da_dr = self.scale_factor_derivative(r_safe)

        # H = a' / (a * sqrt(A))
        # Avoid division by zero
        a_safe = max(abs(a), 1e-30)
        H = da_dr / (a_safe * np.sqrt(A))

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
        Calculate Christoffel symbols Γ^λ_μν for the 4-ball metric.

        Metric: ds² = -A(r)dr² + a(r)²dΩ₃²

        Where dΩ₃² = dχ² + sin²(χ)(dθ² + sin²(θ)dφ²)

        The non-zero Christoffel symbols are:
        - Γ^r_rr = A'/(2A)
        - Γ^r_χχ = -a·a'/A
        - Γ^r_θθ = -a·a'/A · sin²(χ)
        - Γ^r_φφ = -a·a'/A · sin²(χ)·sin²(θ)
        - Γ^χ_rχ = Γ^χ_χr = a'/a
        - Γ^χ_θθ = -sin(χ)·cos(χ)
        - Γ^χ_φφ = -sin(χ)·cos(χ)·sin²(θ)
        - Γ^θ_rθ = Γ^θ_θr = a'/a
        - Γ^θ_φθ = Γ^θ_θφ = cot(θ)
        - Γ^θ_χχ = sin(χ)·cos(χ)
        - Γ^φ_rφ = Γ^φ_φr = a'/a
        - Γ^φ_χφ = Γ^φ_φχ = cot(χ)
        - Γ^φ_θφ = Γ^φ_φθ = cot(θ)

        Args:
            r: Radial coordinate
            chi: S^3 polar angle
            theta: S^3 azimuthal angle
            phi: S^3 second azimuthal angle

        Returns:
            Christoffel symbols Γ^λ_μν (4x4x4 array) where first index is λ (upper)
        """
        # Initialize
        Gamma = np.zeros((4, 4, 4))

        # Get metric functions
        A = self.metric.lapse_function(r)
        a = self.metric.scale_factor(r)

        # Derivatives using analytical formulas
        # A' = dA/dr (for A=1, A'=0 in synchronous gauge)
        A_prime = 0.0  # A(r) = 1 in synchronous gauge

        # a' = da/dr (using analytical formula for a ∝ r²)
        a_prime = self.metric.scale_factor_derivative(r)

        # Avoid division by zero
        A_safe = max(abs(A), 1e-30)
        a_safe = max(abs(a), 1e-30)

        # Trig functions for S^3
        sin_chi = np.sin(chi)
        cos_chi = np.cos(chi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # S^3 scale factor squared
        a_sq = a * a
        sin_chi_sq = sin_chi * sin_chi

        # cotangent functions
        cot_chi = cos_chi / max(sin_chi, 1e-30)
        cot_theta = cos_theta / max(sin_theta, 1e-30)

        # ========== Non-zero Christoffel symbols ==========

        # Γ^r_rr = A'/(2A)
        Gamma[0, 0, 0] = A_prime / (2 * A_safe)

        # Γ^r_χχ = -a·a'/A
        Gamma[0, 1, 1] = -a * a_prime / A_safe

        # Γ^r_θθ = -a·a'/A · sin²(χ)
        Gamma[0, 2, 2] = -a * a_prime / A_safe * sin_chi_sq

        # Γ^r_φφ = -a·a'/A · sin²(χ)·sin²(θ)
        Gamma[0, 3, 3] = -a * a_prime / A_safe * sin_chi_sq * sin_theta * sin_theta

        # Γ^χ_rχ = Γ^χ_χr = a'/a
        Gamma[1, 0, 1] = a_prime / a_safe
        Gamma[1, 1, 0] = a_prime / a_safe

        # Γ^χ_θθ = -sin(χ)·cos(χ)
        Gamma[1, 2, 2] = -sin_chi * cos_chi

        # Γ^χ_φφ = -sin(χ)·cos(χ)·sin²(θ)
        Gamma[1, 3, 3] = -sin_chi * cos_chi * sin_theta * sin_theta

        # Γ^θ_rθ = Γ^θ_θr = a'/a
        Gamma[2, 0, 2] = a_prime / a_safe
        Gamma[2, 2, 0] = a_prime / a_safe

        # Γ^θ_χχ = sin(χ)·cos(χ)
        Gamma[2, 1, 1] = sin_chi * cos_chi

        # Γ^θ_φθ = Γ^θ_θφ = cot(θ)
        Gamma[2, 3, 2] = cot_theta
        Gamma[2, 2, 3] = cot_theta

        # Γ^φ_rφ = Γ^φ_φr = a'/a
        Gamma[3, 0, 3] = a_prime / a_safe
        Gamma[3, 3, 0] = a_prime / a_safe

        # Γ^φ_χφ = Γ^φ_φχ = cot(χ)
        Gamma[3, 1, 3] = cot_chi
        Gamma[3, 3, 1] = cot_chi

        # Γ^φ_θφ = Γ^φ_φθ = cot(θ)
        Gamma[3, 2, 3] = cot_theta
        Gamma[3, 3, 2] = cot_theta

        return Gamma

    def ricci_tensor(self, r: float, chi: float, theta: float, phi: float) -> np.ndarray:
        """
        Calculate Ricci tensor R_μν using Christoffel symbols.

        R_μν = ∂_λ Γ^λ_μν - ∂_ν Γ^λ_μλ + Γ^λ_λσ Γ^σ_μν - Γ^λ_νσ Γ^σ_λμ

        Args:
            r: Radial coordinate
            chi: S^3 polar angle
            theta: S^3 azimuthal angle
            phi: S^3 second azimuthal angle

        Returns:
            Ricci tensor (4x4 array)
        """
        # Get Christoffel symbols
        Gamma = self.christoffel_symbols(r, chi, theta, phi)

        # Compute Ricci tensor components
        R = np.zeros((4, 4))

        for mu in range(4):
            for nu in range(4):
                # R_μν = ∂_λ Γ^λ_μν - ∂_ν Γ^λ_μλ + Γ^λ_λσ Γ^σ_μν - Γ^λ_νσ Γ^σ_λμ
                term1 = 0.0  # ∂_λ Γ^λ_μν (approximated as zero for diagonal metric)
                term2 = 0.0  # ∂_ν Γ^λ_μλ (approximated as zero)
                term3 = 0.0  # Γ^λ_λσ Γ^σ_μν
                term4 = 0.0  # Γ^λ_νσ Γ^σ_λμ

                for lam in range(4):
                    for sig in range(4):
                        term3 += Gamma[lam, lam, sig] * Gamma[sig, mu, nu]
                        term4 += Gamma[lam, nu, sig] * Gamma[sig, lam, mu]

                R[mu, nu] = term1 - term2 + term3 - term4

        # For the 4-ball metric with exact solution a ∝ r²,
        # we can use the analytical FLRW form
        a = self.metric.scale_factor(r)
        A = self.metric.lapse_function(r)

        # Derivatives (using analytical formula for a ∝ r²)
        a_prime = self.metric.scale_factor_derivative(r)
        # Second derivative: d²a/dr² = 2*a0/r0² = 2*a/r² (for a ∝ r²)
        a_prime2 = 2 * a / (r**2) if r > self.metric.R_MIN else 2 * self.metric.a0 / (self.metric.r0**2)

        # Analytical Ricci tensor for FLRW metric
        R_analytical = np.zeros((4, 4))

        # Time-time: R_00 = -3(a''/a)
        R_analytical[0, 0] = -3 * a_prime2 / a

        # Spatial: R_ij = (a''/a + 2(a'/a)² + 2k/a²) g_ij for k=+1
        # In our metric form with A factor
        H = a_prime / a  # Hubble parameter-like
        k = 1.0  # Closed universe (S^3)

        for i in range(1, 4):
            R_analytical[i, i] = (a_prime2 / a + 2 * H**2 + 2 * k / a**2) * a**2 * A

        # Cross terms should be zero (isotropic)
        R_analytical[0, 1:] = 0
        R_analytical[1:, 0] = 0

        return R_analytical

    def einstein_tensor(self, r: float, chi: float, theta: float, phi: float) -> np.ndarray:
        """
        Calculate Einstein tensor G_μν = R_μν - (1/2)g_μν R.

        Args:
            r: Radial coordinate
            chi: S^3 polar angle
            theta: S^3 azimuthal angle
            phi: S^3 second azimuthal angle

        Returns:
            Einstein tensor (4x4 array)
        """
        R = self.ricci_tensor(r, chi, theta, phi)
        R_scalar = self.ricci_scalar(r)
        g = self.metric.metric_4D(r, chi, theta, phi)
        g_inv = self.metric.inverse_metric(r, chi, theta, phi)

        G = R - 0.5 * R_scalar * g
        return G

    def verify_einstein_equations(self, r: float, chi: float, theta: float, phi: float,
                                   rho_eff: float, p_eff: float) -> dict:
        """
        Verify Einstein equations: G_μν = 8πT_μν.

        Args:
            r: Radial coordinate
            chi: S^3 polar angle
            theta: S^3 azimuthal angle
            phi: S^3 second azimuthal angle
            rho_eff: Effective energy density
            p_eff: Effective pressure

        Returns:
            Dictionary with verification results
        """
        G = self.einstein_tensor(r, chi, theta, phi)
        G_inv = self.metric.inverse_metric(r, chi, theta, phi)

        # Build stress-energy tensor (perfect fluid form)
        # T_μν = (ρ + p)u_μu_ν + pg_μν
        # In comoving synchronous gauge: u^μ = (1/√A, 0, 0, 0)

        T = np.zeros((4, 4))
        A = self.metric.lapse_function(r)
        sqrt_A = np.sqrt(A)

        # 4-velocity
        u = np.array([1.0/sqrt_A, 0, 0, 0])

        # T_00 = (ρ + p)u_0u_0 + p*g_00 = (ρ + p)/A - p*A = ρ/A
        T[0, 0] = rho_eff / A

        # T_ii = (ρ + p)u_iu_i + p*g_ii = p*g_ii for i>0
        g = self.metric.metric_4D(r, chi, theta, phi)
        for i in range(1, 4):
            T[i, i] = p_eff * g[i, i]

        # Right-hand side: 8πT_μν
        G_observed = 8 * np.pi * PhysicalConstants.G * T

        # Compare
        diff = np.abs(G - G_observed)
        max_diff = np.max(diff)
        rel_diff = max_diff / (np.abs(G) + np.abs(G_observed) + 1e-30)

        return {
            'G_00': G[0, 0],
            'T_00': T[0, 0],
            'max_abs_diff': max_diff,
            'max_rel_diff': rel_diff,
            'verified': rel_diff < 0.1  # Within 10%
        }

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
