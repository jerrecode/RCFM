#!/usr/bin/env python3
"""
RCFM Core Module - Dual-Stream Fluid Dynamics

This module implements the dual-stream energy-momentum tensor
and fluid dynamics for the RCFM.

Author: MiniMax Agent
Version: 1.3
"""

import numpy as np
from typing import Tuple, Optional
from scipy import integrate
from .constants import PhysicalConstants, RCFMParameters
from .metric import MetricRCFM


class DualStreamFluid:
    """
    Dual-Stream Energy-Momentum Tensor

    Implements the RCFM dual-stream fluid with Phase A (active matter)
    and Phase B (ghost quanta).
    """

    def __init__(self, params: RCFMParameters):
        """
        Initialize dual-stream fluid.

        Args:
            params: RCFMParameters instance
        """
        self.params = params
        self.c = PhysicalConstants.c

    def rho_A(self, r: float, rho_A0: float = None) -> float:
        """
        Calculate Phase A density at radial coordinate r.

        Args:
            r: Radial coordinate
            rho_A0: Phase A density at present (optional)

        Returns:
            Phase A density rho_A(r)
        """
        metric = MetricRCFM(self.params)
        a = metric.scale_factor(r)

        if rho_A0 is None:
            # Use critical density as reference
            rho_A0 = PhysicalConstants.rho_crit_0

        # Matter-dominated: rho_A ∝ a^(-3)
        rho_A = rho_A0 * a**(-3)

        return rho_A

    def rho_B(self, r: float, rho_B0: float = None) -> float:
        """
        Calculate Phase B density at radial coordinate r.

        Phase B decays due to drag interaction.

        Args:
            r: Radial coordinate
            rho_B0: Phase B density at present (optional)

        Returns:
            Phase B density rho_B(r)
        """
        if rho_B0 is None:
            rho_B0 = self.params.rho_B0

        # Phase B decays exponentially due to drag
        decay = np.exp(-self.params.Gamma_drag * r)
        rho_B = rho_B0 * decay

        return rho_B

    def p_A(self, r: float) -> float:
        """
        Calculate Phase A pressure.

        Args:
            r: Radial coordinate

        Returns:
            Phase A pressure p_A(r)
        """
        w = self.equation_of_state(r)
        rho_A = self.rho_A(r)
        return w * rho_A

    def equation_of_state(self, r: float) -> float:
        """
        Calculate Phase A equation of state parameter w.

        w = p/rho for different cosmic epochs:
        - Radiation: w = 1/3
        - Matter: w = 0

        Args:
            r: Radial coordinate

        Returns:
            Equation of state parameter w
        """
        metric = MetricRCFM(self.params)
        a = metric.scale_factor(r)

        # Simple transition: w = 1/3 for a < a_eq, w = 0 for a > a_eq
        a_eq = 1.0 / 3400  # Matter-radiation equality scale factor

        if a < a_eq:
            w = 1.0 / 3.0  # Radiation
        else:
            w = 0.0  # Matter

        return w

    def energy_momentum_tensor(self, r: float) -> dict:
        """
        Calculate total energy-momentum tensor components.

        Args:
            r: Radial coordinate

        Returns:
            Dictionary with T^μ_ν components
        """
        rho_A = self.rho_A(r)
        rho_B = self.rho_B(r)
        p_A = self.p_A(r)

        # Effective density and pressure
        beta = self.params.beta
        rho_eff = rho_A + rho_B + 2 * beta * rho_A * rho_B
        p_eff = p_A - beta * self.params.Gamma_drag * rho_A * rho_B * (rho_A - rho_B) / (3 * self.params.Lambda_RCFM(0))

        return {
            'rho_A': rho_A,
            'rho_B': rho_B,
            'p_A': p_A,
            'p_B': 0,  # Pressureless
            'rho_eff': rho_eff,
            'p_eff': p_eff
        }

    def drag_force(self, r: float) -> float:
        """
        Calculate drag force between Phase A and Phase B.

        Args:
            r: Radial coordinate

        Returns:
            Drag force per unit volume
        """
        rho_A = self.rho_A(r)
        rho_B = self.rho_B(r)
        Gamma = self.params.Gamma_drag

        # Drag force density
        f_drag = Gamma * rho_A * rho_B

        return f_drag


class FriedmannSolver:
    """
    Friedmann Equation Solver for RCFM

    Solves the modified Friedmann equations numerically.
    """

    def __init__(self, params: RCFMParameters):
        """
        Initialize solver.

        Args:
            params: RCFMParameters instance
        """
        self.params = params
        self.fluid = DualStreamFluid(params)

    def friedmann_rhs(self, r: float, y: np.ndarray) -> np.ndarray:
        """
        Right-hand side of Friedmann equations.

        dy/dr = f(y, r)

        where y = [a, H, rho_A, rho_B]

        Args:
            r: Radial coordinate
            y: State vector [a, H, rho_A, rho_B]

        Returns:
            Derivatives [da/dr, dH/dr, drho_A/dr, drho_B/dr]
        """
        a, H, rho_A, rho_B = y
        beta = self.params.beta
        Gamma = self.params.Gamma_drag

        # Effective density
        rho_eff = rho_A + rho_B + 2 * beta * rho_A * rho_B

        # Equation of state
        w = self.fluid.equation_of_state(r)
        p_A = w * rho_A

        # Effective pressure
        p_eff = p_A - beta * Gamma * rho_A * rho_B * (rho_A - rho_B) / (3 * H)

        # Friedmann equation: H^2 = (8πG/3) rho_eff - 1/a^2
        G = PhysicalConstants.G
        H_from_Friedmann = np.sqrt((8 * np.pi * G / 3) * rho_eff - 1 / a**2)

        # da/dr = a * H (using H from Friedmann)
        da_dr = a * H_from_Friedmann

        # dH/dr from Raychaudhuri
        dH_dr = (4 * np.pi * G / 3) * (rho_eff + 3 * p_eff) - H**2

        # Continuity equations with drag
        drho_A_dr = -H * (rho_A + p_A) + Gamma * rho_A * rho_B
        drho_B_dr = H * rho_B - Gamma * rho_A * rho_B

        return np.array([da_dr, dH_dr, drho_A_dr, drho_B_dr])

    def solve(self,
              r_start: float,
              r_end: float,
              initial_conditions: np.ndarray,
              r_step: float = 1e-6) -> dict:
        """
        Solve Friedmann equations.

        Args:
            r_start: Starting radial coordinate
            r_end: Ending radial coordinate
            initial_conditions: Initial state [a, H, rho_A, rho_B]
            r_step: Integration step size

        Returns:
            Dictionary with solution arrays
        """
        from scipy.integrate import solve_ivp

        # Create radial coordinate array
        r_array = np.arange(r_start, r_end, r_step)

        # Solve ODE
        solution = solve_ivp(
            self.friedmann_rhs,
            [r_start, r_end],
            initial_conditions,
            method='RK45',
            t_eval=r_array,
            rtol=1e-8,
            atol=1e-10
        )

        return {
            'r': solution.t,
            'a': solution.y[0, :],
            'H': solution.y[1, :],
            'rho_A': solution.y[2, :],
            'rho_B': solution.y[3, :]
        }


def compute_perturbation_growth(params: RCFMParameters,
                                k: float,
                                z: float) -> float:
    """
    Compute matter perturbation growth factor.

    Args:
        params: RCFM parameters
        k: Wavenumber (h/Mpc)
        z: Redshift

    Returns:
        Growth factor D(z)
    """
    # LCDM growth factor (approximation)
    a = 1.0 / (1 + z)
    D_LCDM = a

    # RCFM modification
    Lambda_RCFM = params.Lambda_RCFM(z)
    D_RCFM = D_LCDM * (1 + 3/5 * Lambda_RCFM)

    return D_RCFM


if __name__ == "__main__":
    # Test the module
    print("RCFM Dual-Stream Fluid Dynamics")
    print("=" * 50)

    from .constants import get_rcfm_defaults

    params = get_rcfm_defaults()
    fluid = DualStreamFluid(params)

    # Test at present
    r_present = params.Rmax
    print(f"At present (r = {r_present:.2e} m):")
    emt = fluid.energy_momentum_tensor(r_present)
    print(f"  rho_A = {emt['rho_A']:.2e} kg/m^3")
    print(f"  rho_B = {emt['rho_B']:.2e} kg/m^3")
    print(f"  p_A = {emt['p_A']:.2e} kg/m^3")
    print(f"  rho_eff = {emt['rho_eff']:.2e} kg/m^3")
    print(f"  p_eff = {emt['p_eff']:.2e} kg/m^3")
    print(f"  Drag force = {fluid.drag_force(r_present):.2e}")
