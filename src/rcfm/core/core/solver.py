#!/usr/bin/env python3
"""
RCFM Core Module - ODE Solver for Background Evolution

This module implements the ODE solver for the RCFM background
evolution equations.

Author: MiniMax Agent
Version: 1.3
"""

import numpy as np
from typing import Tuple, Optional, Callable
from scipy.integrate import solve_ivp, odeint
from scipy.interpolate import interp1d
from .constants import PhysicalConstants, RCFMParameters
from .fluids import DualStreamFluid


class BackgroundSolver:
    """
    Solver for RCFM Background Evolution

    Solves the system of ODEs:
    - da/dr = sqrt(A) * H * a
    - dH/dr = sqrt(A) * [4πG/3 * (ρ_eff + 3p_eff) - H²]
    - dρA/dr = -sqrt(A) * H * (ρA + pA) + Γ * ρA * ρB
    - dρB/dr = sqrt(A) * H * ρB - Γ * ρA * ρB
    """

    def __init__(self, params: RCFMParameters):
        """
        Initialize background solver.

        Args:
            params: RCFMParameters instance
        """
        self.params = params
        self.fluid = DualStreamFluid(params)
        self.G = PhysicalConstants.G

    def rhs(self, r: float, y: np.ndarray) -> np.ndarray:
        """
        Right-hand side of background ODEs.

        Args:
            r: Radial coordinate
            y: State vector [a, H, rho_A, rho_B]

        Returns:
            Derivatives [da/dr, dH/dr, drho_A/dr, drho_B/dr]
        """
        a, H, rho_A, rho_B = y

        # Get equation of state
        w = self.fluid.equation_of_state(r)
        p_A = w * rho_A

        # Effective quantities (per paper equations 18-22)
        beta = self.params.beta
        Gamma = self.params.Gamma_drag
        rho_eff = rho_A + rho_B + 2 * beta * rho_A * rho_B

        # Effective pressure: corrected formula (BUG-FIX-2)
        # Paper eq (22): p_eff = p_A - (βΓ/3H) * ρ_A * ρ_B * (ρ_A - ρ_B)
        # Using H variable instead of Lambda_RCFM(0)
        # Handle H → 0 case to avoid division by zero
        H_safe = max(abs(H), 1e-30) * np.sign(H) if H != 0 else 1e-30
        p_eff = p_A - (beta * Gamma * rho_A * rho_B * (rho_A - rho_B)) / (3 * H_safe)

        # Lapse function (synchronous gauge)
        A = 1.0
        sqrt_A = np.sqrt(A)

        # da/dr = sqrt(A) * H * a
        da_dr = sqrt_A * H * a

        # Raychaudhuri equation (BUG-FIX-1): CORRECTED SIGN
        # Paper eq (28): H'/√A + H² = -(4πG/3)(ρ_eff + 3p_eff)
        # Rearranged: dH/dr = √A * [-(4πG/3)(ρ_eff + 3p_eff) - H²]
        dH_dr = sqrt_A * (-(4 * np.pi * self.G / 3) * (rho_eff + 3 * p_eff) - H**2)

        # drho_A/dr = -sqrt(A) * H * (rho_A + p_A) + Gamma * rho_A * rho_B
        drho_A_dr = -sqrt_A * H * (rho_A + p_A) + Gamma * rho_A * rho_B

        # drho_B/dr = sqrt(A) * H * rho_B - Gamma * rho_A * rho_B
        drho_B_dr = sqrt_A * H * rho_B - Gamma * rho_A * rho_B

        return np.array([da_dr, dH_dr, drho_A_dr, drho_B_dr])

    def solve(self,
              r_start: float,
              r_end: float,
              a_init: float = 1e-10,
              H_init: float = None,
              rho_A_init: float = None,
              rho_B_init: float = None,
              r_step: float = 1e-6,
              method: str = 'RK45') -> dict:
        """
        Solve background evolution equations.

        Args:
            r_start: Starting radial coordinate (usually near singularity)
            r_end: Ending radial coordinate (usually present day, Rmax)
            a_init: Initial scale factor
            H_init: Initial Hubble parameter (if None, computed from Friedmann)
            rho_A_init: Initial Phase A density (if None, uses Planck)
            rho_B_init: Initial Phase B density (if None, uses params.rho_B0)
            r_step: Integration step size
            method: Integration method ('RK45', 'RK23', 'Radau')

        Returns:
            Dictionary with solution arrays
        """
        # Set initial conditions
        if H_init is None:
            # Use Friedmann equation to set initial H
            rho_A_init = rho_A_init if rho_A_init else PhysicalConstants.rho_crit_0
            rho_B_init = rho_B_init if rho_B_init else self.params.rho_B0
            H_init = np.sqrt(8 * np.pi * self.G / 3 * rho_A_init)

        if rho_A_init is None:
            rho_A_init = PhysicalConstants.rho_crit_0

        if rho_B_init is None:
            rho_B_init = self.params.rho_B0

        y0 = np.array([a_init, H_init, rho_A_init, rho_B_init])

        # Solve ODE
        solution = solve_ivp(
            lambda r, y: self.rhs(r, y),
            [r_start, r_end],
            y0,
            method=method,
            t_eval=np.linspace(r_start, r_end, 10000),
            rtol=1e-8,
            atol=1e-12,
            dense_output=True
        )

        # BUG-FIX-7: Check solution success and handle failures
        if not solution.success:
            # Log warning but don't crash - allow calling code to handle
            import warnings
            warnings.warn(
                f"ODE solver failed: {solution.message}. "
                f"Status: {solution.status}. Results may be unreliable.",
                RuntimeWarning
            )

        # Verify solution contains valid data
        if len(solution.t) == 0:
            raise RuntimeError(
                f"ODE solver produced no output points. "
                f"Initial conditions may be invalid: y0={y0}"
            )

        # Check for NaN or Inf in solution
        y_solution = solution.y
        if np.any(~np.isfinite(y_solution)):
            import warnings
            warnings.warn(
                "Solution contains NaN or Inf values. "
                "Check initial conditions and physics parameters.",
                RuntimeWarning
            )

        return {
            'r': solution.t,
            'a': solution.y[0, :],
            'H': solution.y[1, :],
            'rho_A': solution.y[2, :],
            'rho_B': solution.y[3, :],
            'success': solution.success,
            'message': solution.message,
            'status': solution.status,
            'nfev': solution.nfev,
            'njev': solution.njev
        }

    def compute_H_of_z(self, solution: dict) -> Callable:
        """
        Compute H(z) from solution.

        Args:
            solution: Solution dictionary from solve()

        Returns:
            Function H(z)
        """
        r = solution['r']
        a = solution['a']
        H = solution['H']

        # Convert to redshift
        z = 1.0 / a - 1.0

        # Create interpolator
        H_interp = interp1d(z, H, kind='cubic', fill_value='extrapolate')

        return H_interp

    def compute_distance_modulus(self, solution: dict, z: float) -> float:
        """
        Compute distance modulus.

        Args:
            solution: Solution dictionary
            z: Redshift

        Returns:
            Distance modulus
        """
        r = solution['r']
        a = solution['a']

        # Comoving distance
        # d_c = ∫ dz/H(z) [simplified]
        chi = np.trapz(1.0 / solution['H'], r)

        # Luminosity distance
        d_L = chi * (1 + z)

        # Distance modulus
        mu = 5 * np.log10(d_L / (10 * parsec))

        return mu


def solve_background(params: RCFMParameters,
                    r_start: float = 1e-10,
                    r_end: float = None) -> dict:
    """
    Convenience function to solve background equations.

    Args:
        params: RCFM parameters
        r_start: Starting radial coordinate
        r_end: Ending radial coordinate (default: params.Rmax)

    Returns:
        Solution dictionary
    """
    if r_end is None:
        r_end = params.Rmax

    solver = BackgroundSolver(params)

    # Initial conditions near singularity
    # At r → 0: a ∝ r², so take small r
    r_init = r_start

    # Initial scale factor
    a_init = (r_init / r_end)**2

    # Initial Hubble parameter from Friedmann
    G = PhysicalConstants.G
    rho_init = PhysicalConstants.rho_crit_0 * (r_init / r_end)**(-3)
    H_init = np.sqrt(8 * np.pi * G / 3 * rho_init)

    # Initial densities
    rho_A_init = rho_init
    rho_B_init = params.rho_B0 * np.exp(-params.Gamma_drag * r_init)

    return solver.solve(
        r_start=r_init,
        r_end=r_end,
        a_init=a_init,
        H_init=H_init,
        rho_A_init=rho_A_init,
        rho_B_init=rho_B_init
    )


# Constants
parsec = 3.086e16  # meters


if __name__ == "__main__":
    print("RCFM Background Solver Test")
    print("=" * 50)

    from constants import get_rcfm_defaults

    params = get_rcfm_defaults()
    solution = solve_background(params)

    print(f"Solution status: {solution['success']}")
    print(f"Message: {solution['message']}")
    print(f"Number of points: {len(solution['r'])}")
    print()
    print(f"Initial (r = {solution['r'][0]:.2e} m):")
    print(f"  a = {solution['a'][0]:.2e}")
    print(f"  H = {solution['H'][0]:.2e} 1/s")
    print()
    print(f"Final (r = {solution['r'][-1]:.2e} m):")
    print(f"  a = {solution['a'][-1]:.2e}")
    print(f"  H = {solution['H'][-1]:.2e} 1/s")
    print(f"  H = {solution['H'][-1] * 3.086e22 / 1000:.2f} km/s/Mpc")
