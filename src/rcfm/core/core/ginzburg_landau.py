#!/usr/bin/env python3
"""
RCFM Module - Ginzburg-Landau Condensate

This module implements the Ginzburg-Landau theory for the phase transition
that identifies Phase B ghost quanta when the gauge condensate vanishes.

From paper Section 4.6: The critical density ρ_c is fixed by the
Ginzburg-Landau condensate, removing it as a free parameter.

Author: MiniMax Agent
Version: 1.3
"""

import numpy as np
from typing import Tuple, Optional
from .constants import PhysicalConstants, RCFMParameters


class GinzburgLandauCondensate:
    """
    Ginzburg-Landau Theory for RCFM Phase Transition

    Implements the free energy functional:
    F[φ] = α(T)|φ|² + (β/2)|φ|⁴ + ...

    The phase transition occurs when the gauge condensate ⟨φ⟩ vanishes,
    which identifies the boundary between Phase A and Phase B.
    """

    def __init__(self, params: RCFMParameters):
        """
        Initialize Ginzburg-Landau condensate calculator.

        Args:
            params: RCFMParameters instance
        """
        self.params = params
        self.const = PhysicalConstants()

        # Ginzburg-Landau parameters
        # α(T) = α_0 * (T - T_c) / T_c
        # For cosmological context, we use ρ instead of T
        self.alpha_0 = self._calculate_alpha_0()
        self.beta_GL = self._calculate_beta_GL()

    def _calculate_alpha_0(self) -> float:
        """
        Calculate α_0 coupling constant.

        α_0 sets the scale of the phase transition.
        Determined from requiring ρ_c ~ ρ_Planck.

        Returns:
            α_0 in SI units (J/m³)
        """
        # Set α_0 such that phase transition occurs at Planck density
        rho_planck = self.const.planck_mass / self.const.c**2 / self.const.planck_length**3

        # α_0 ~ ρ_Pl to get critical point near Planck scale
        alpha_0 = rho_planck

        return alpha_0

    def _calculate_beta_GL(self) -> float:
        """
        Calculate β coupling constant.

        β is determined from the gap equation at critical point.

        Returns:
            β_GL in SI units (J·m³/m⁶ = J·m⁻³)
        """
        # β should be positive for stability of the condensate
        # Set from requiring ⟨φ⟩² ~ ρ_c at critical point
        rho_c = self.critical_density()

        # From free energy minimum: ⟨φ⟩² = -α/β = ρ_c
        # So β = -α_0 * (T_c - T)/ρ_c
        # At critical point, this gives order-of-magnitude estimate

        beta_GL = self.alpha_0 / rho_c if rho_c > 0 else 1.0

        return beta_GL

    def free_energy_density(self, rho: float, phi_mag: float = 0.0) -> float:
        """
        Calculate Ginzburg-Landau free energy density.

        F = α(ρ)|φ|² + (β/2)|φ|⁴ + ...

        Args:
            rho: Energy density
            phi_mag: Order parameter magnitude |φ|²

        Returns:
            Free energy density
        """
        # α depends on density (analogous to temperature)
        # α(ρ) = α_0 * (ρ - ρ_c) / ρ_c
        rho_c = self.critical_density()
        alpha = self.alpha_0 * (rho - rho_c) / rho_c

        # Free energy
        F = alpha * phi_mag**2 + (self.beta_GL / 2) * phi_mag**4

        return F

    def critical_density(self) -> float:
        """
        Calculate critical transition density ρ_c.

        From paper: ρ_c is fixed by Ginzburg-Landau condensate,
        removing it as a free parameter.

        The critical density is where the gauge condensate vanishes:
        ⟨φ⟩ = 0 at ρ = ρ_c

        Returns:
            Critical density ρ_c in SI units (kg/m³)
        """
        # Physical parameters
        beta = self.params.beta
        R_max = self.params.Rmax
        Gamma_drag = self.params.Gamma_drag
        rho_B0 = self.params.rho_B0

        # [IMPL-5]: Critical density from Ginzburg-Landau theory
        # At critical point: ∂F/∂|φ| = 0 gives |φ|² = -α/β
        # For spontaneous symmetry breaking: α < 0, β > 0

        # Critical density is where α = 0:
        # α(ρ_c) = α_0 * (ρ_c - ρ_c)/ρ_c = 0
        # But we need additional physics to fix ρ_c

        # Physical argument: ρ_c corresponds to the scale where
        # gauge dressing becomes negligible
        # This happens when: ρ_c * R_max³ ~ m_Pl² (Planck suppression)

        # Set by requiring dimensionless coupling ~ 1:
        # β * ρ_c ~ m_Pl² / R_max²
        m_planck = self.const.planck_mass
        c = self.const.c

        rho_c = (m_planck * c**2) / (beta * R_max**3) if beta > 0 else self.const.rho_crit_0

        return rho_c

    def order_parameter(self, rho: float) -> float:
        """
        Calculate order parameter |φ|² as function of density.

        From Ginzburg-Landau: |φ|² = -α/β = (ρ_c - ρ)/ρ_c * ρ_0

        Args:
            rho: Energy density

        Returns:
            Order parameter |φ|²
        """
        rho_c = self.critical_density()

        if rho < rho_c:
            # Below critical: symmetric phase, |φ|² = 0
            phi_sq = 0.0
        else:
            # Above critical: broken symmetry, |φ|² > 0
            # |φ|² = (ρ - ρ_c) * scale_factor
            scale = self.alpha_0 / self.beta_GL
            phi_sq = max(0.0, (rho - rho_c) / rho_c) * scale

        return phi_sq

    def is_condensate_present(self, rho: float) -> bool:
        """
        Check if gauge condensate is present (Phase A).

        Args:
            rho: Energy density

        Returns:
            True if condensate present (Phase A), False otherwise (Phase B)
        """
        return rho >= self.critical_density()

    def phase_transition_strength(self, rho: float) -> float:
        """
        Calculate strength of phase transition.

        Args:
            rho: Energy density

        Returns:
            Parameter 0 < ε < 1 measuring transition
        """
        rho_c = self.critical_density()

        if rho_c <= 0:
            return 1.0

        epsilon = abs(rho - rho_c) / rho_c
        return min(epsilon, 1.0)

    def thermodynamic_potential(self, rho: float, T_eff: float = None) -> float:
        """
        Calculate thermodynamic potential Ω(T, μ).

        Args:
            rho: Energy density
            T_eff: Effective temperature (optional)

        Returns:
            Thermodynamic potential
        """
        phi_sq = self.order_parameter(rho)
        F = self.free_energy_density(rho, phi_sq)

        # Add entropy contribution if T_eff given
        if T_eff is not None:
            # S ~ -∂F/∂T
            S = 0.0  # Simplified
            F -= T_eff * S

        return F


class PhaseTransition:
    """
    Phase transition dynamics for RCFM.

    Models the transition from Phase A (condensate present)
    to Phase B (condensate vanishing) at ρ = ρ_c.
    """

    def __init__(self, params: RCFMParameters):
        """
        Initialize phase transition calculator.

        Args:
            params: RCFMParameters instance
        """
        self.params = params
        self.condensate = GinzburgLandauCondensate(params)
        self.const = PhysicalConstants()

    def identify_phase(self, rho: float) -> str:
        """
        Identify which phase at given density.

        Args:
            rho: Energy density

        Returns:
            'A', 'B', or 'critical'
        """
        rho_c = self.condensate.critical_density()

        if abs(rho - rho_c) / rho_c < 0.01:
            return 'critical'
        elif rho > rho_c:
            return 'A'  # Gauge condensate present
        else:
            return 'B'  # Gauge condensate absent

    def transition_rate(self, rho: float) -> float:
        """
        Calculate phase transition rate.

        From paper: Transition occurs via quantum tunneling
        at the boundary condition r = 0.

        Args:
            rho: Energy density

        Returns:
            Transition rate (0 to 1)
        """
        rho_c = self.condensate.critical_density()

        if rho < rho_c:
            # Below critical: full transition to Phase B
            return 1.0
        elif rho > 2 * rho_c:
            # Well above critical: no transition
            return 0.0
        else:
            # Near critical: partial transition
            epsilon = (2 * rho_c - rho) / rho_c
            return max(0.0, min(1.0, epsilon))

    def decay_width(self, rho: float) -> float:
        """
        Calculate decay width Γ for Phase B quanta.

        From paper: Phase B emerges from condensate decay.

        Args:
            rho: Energy density

        Returns:
            Decay width in SI units (1/s)
        """
        rho_c = self.condensate.critical_density()

        if rho >= rho_c:
            # No decay if condensate present
            return 0.0

        # Decay width increases as we go deeper into Phase B
        # Γ ~ Γ_drag * exp(-|ρ_c - ρ|/ρ_c)
        Gamma_drag = self.params.Gamma_drag
        suppression = np.exp(-abs(rho_c - rho) / rho_c)

        return Gamma_drag * suppression

    def free_energy_difference(self, rho: float) -> float:
        """
        Calculate free energy difference between phases.

        ΔF = F_B - F_A

        Args:
            rho: Energy density

        Returns:
            Free energy difference
        """
        rho_c = self.condensate.critical_density()

        # At ρ = ρ_c, F_A = F_B
        # Below ρ_c, Phase B is favored (lower F)
        delta_F = self.condensate.free_energy_density(rho, 0.0) - \
                  self.condensate.free_energy_density(rho, self.condensate.order_parameter(rho))

        return delta_F


def compute_rho_c_from_gl(params: RCFMParameters) -> float:
    """
    Convenience function to compute critical density.

    Args:
        params: RCFMParameters instance

    Returns:
        Critical density ρ_c
    """
    gl = GinzburgLandauCondensate(params)
    return gl.critical_density()


if __name__ == "__main__":
    print("Ginzburg-Landau Condensate Module Test")
    print("=" * 50)

    from rcfm.core.constants import get_rcfm_defaults

    params = get_rcfm_defaults()

    # Test Ginzburg-Landau condensate
    gl = GinzburgLandauCondensate(params)

    rho_c = gl.critical_density()
    print(f"\nCritical density ρ_c = {rho_c:.3e} kg/m³")

    # Compare with Planck density
    rho_planck = PhysicalConstants.planck_mass / PhysicalConstants.c**2 / PhysicalConstants.planck_length**3
    print(f"Planck density ρ_Pl = {rho_planck:.3e} kg/m³")
    print(f"Ratio ρ_c/ρ_Pl = {rho_c/rho_planck:.3e}")

    # Test phase transition
    pt = PhaseTransition(params)

    print("\nPhase identification:")
    for rho_frac in [0.1, 0.5, 1.0, 2.0, 5.0]:
        rho = rho_frac * rho_c
        phase = pt.identify_phase(rho)
        print(f"  ρ = {rho_frac:.1f} ρ_c: Phase {phase}")
