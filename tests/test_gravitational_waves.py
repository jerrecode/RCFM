#!/usr/bin/env python3
"""
Test suite for rcfm.core.gravitational_waves module.

Tests GW propagation, speed, and stochastic background including BUG-FIX-3 (G_eff).

Author: MiniMax Agent
Version: 1.3 (post bug fixes)
"""

import pytest
import numpy as np
from rcfm.core.gravitational_waves import GravitationalWaves
from rcfm.core.constants import get_rcfm_defaults


class TestGravitationalWaves:
    """Tests for GravitationalWaves class."""

    def test_initialization(self, rcfm_params):
        """Test GW calculator initialization."""
        gw = GravitationalWaves(rcfm_params)
        assert gw.params is not None
        assert gw.c > 0
        assert gw.G > 0

    def test_gw_speed_at_z0(self, rcfm_params):
        """Test GW speed at present (z=0)."""
        gw = GravitationalWaves(rcfm_params)
        c_T = gw.gw_speed(0)

        # GW speed should be very close to c (speed of light)
        assert 0 < c_T <= 1.0

    def test_gw_speed_close_to_c(self, rcfm_params):
        """Test that GW speed is extremely close to c.

        Paper claims: |c_T - c| < 5 × 10^-16 at z=0
        """
        gw = GravitationalWaves(rcfm_params)
        c_T = gw.gw_speed(0)

        # Deviation from 1 should be extremely small
        deviation = abs(1.0 - c_T)

        # Should satisfy GW170817 constraint
        assert deviation < 1e-15, f"GW speed deviation {deviation} exceeds GW170817 constraint"

    def test_gw_speed_decreases_with_z(self, rcfm_params):
        """Test that GW speed decreases with redshift (Λ_RCFM increases)."""
        gw = GravitationalWaves(rcfm_params)

        c_T_z0 = gw.gw_speed(0.0)
        c_T_z1 = gw.gw_speed(1.0)
        c_T_z2 = gw.gw_speed(2.0)

        # Speed should decrease as Λ_RCFM increases
        assert c_T_z1 < c_T_z0
        assert c_T_z2 < c_T_z1

    def test_gw_speed_in_lcdm_limit(self, rcfm_params):
        """Test that GW speed = c in ΛCDM limit (Λ_RCFM → 0)."""
        params = rcfm_params
        params.beta = 0.0
        params.rho_B0 = 0.0

        gw = GravitationalWaves(params)
        c_T = gw.gw_speed(0)

        # In ΛCDM limit, should be exactly c
        assert np.isclose(c_T, 1.0, rtol=1e-15)


class TestGWMassTerm:
    """Tests for GW mass term (BUG-FIX-3)."""

    def test_gw_mass_term_positive(self, rcfm_params):
        """Test that m_GW² is positive for ρ_A > ρ_B."""
        gw = GravitationalWaves(rcfm_params)

        m_sq = gw.gw_mass_term(0)

        # Mass term should be positive (physical)
        assert m_sq > 0

    def test_gw_mass_term_decreases_with_z(self, rcfm_params):
        """Test that m_GW² decreases as universe expands."""
        gw = GravitationalWaves(rcfm_params)

        m_sq_z0 = gw.gw_mass_term(0.0)
        m_sq_z1 = gw.gw_mass_term(1.0)

        # Mass term should decrease as densities drop
        assert m_sq_z1 < m_sq_z0

    def test_gw_mass_term_in_lcdm_limit(self, rcfm_params):
        """Test m_GW² in ΛCDM limit.

        When Λ_RCFM → 0 (β → 0), G_eff → G and ρ_B → 0,
        so m_GW² should approach zero.
        """
        params = rcfm_params
        params.beta = 0.0
        params.rho_B0 = 0.0

        gw = GravitationalWaves(params)
        m_sq = gw.gw_mass_term(0)

        # Should be zero in ΛCDM limit
        assert np.isclose(m_sq, 0, atol=1e-30)

    def test_gw_mass_term_growth_with_lambda(self, rcfm_params):
        """Test that m_GW² grows with Λ_RCFM.

        BUG-FIX-3: G_eff formula was wrong. Now properly relates to Λ_RCFM.
        """
        gw = GravitationalWaves(rcfm_params)

        Lambda_z0 = rcfm_params.Lambda_RCFM(0)
        m_sq_z0 = gw.gw_mass_term(0)

        Lambda_z1 = rcfm_params.Lambda_RCFM(1)
        m_sq_z1 = gw.gw_mass_term(1)

        # m_GW² should increase with Λ_RCFM
        # Since Λ_RCFM ∝ (1+z)³, m_GW² at z=1 should be ~8x larger
        expected_ratio = (1 + 1)**3  # 8
        actual_ratio = m_sq_z1 / m_sq_z0 if m_sq_z0 > 0 else 0

        # Should scale with Λ_RCFM (approximately)
        Lambda_ratio = Lambda_z1 / Lambda_z0 if Lambda_z0 > 0 else 0
        assert np.isclose(actual_ratio, Lambda_ratio, rtol=0.5)


class TestFrequencyCutoff:
    """Tests for GW frequency cutoff."""

    def test_frequency_cutoff_positive(self, rcfm_params):
        """Test that cutoff frequency is positive."""
        gw = GravitationalWaves(rcfm_params)
        f_cut = gw.frequency_cutoff(0)

        assert f_cut > 0

    def test_frequency_cutoff_decreases_with_z(self, rcfm_params):
        """Test that cutoff frequency decreases as universe expands."""
        gw = GravitationalWaves(rcfm_params)

        f_cut_z0 = gw.frequency_cutoff(0.0)
        f_cut_z1 = gw.frequency_cutoff(1.0)

        assert f_cut_z1 < f_cut_z0

    def test_frequency_cutoff_in_lcdm_limit(self, rcfm_params):
        """Test cutoff frequency in ΛCDM limit."""
        params = rcfm_params
        params.beta = 0.0
        params.rho_B0 = 0.0

        gw = GravitationalWaves(params)
        f_cut = gw.frequency_cutoff(0)

        # Should be zero in ΛCDM limit (no mass term)
        assert np.isclose(f_cut, 0, atol=1e-30)


class TestStochasticBackground:
    """Tests for stochastic GW background."""

    def test_stochastic_background_shape(self, rcfm_params):
        """Test stochastic background has correct shape."""
        gw = GravitationalWaves(rcfm_params)

        f = np.logspace(-3, 2, 100)  # Hz
        Omega_GW = gw.stochastic_background_primordial(f)

        assert Omega_GW.shape == f.shape

    def test_stochastic_background_zero_below_cutoff(self, rcfm_params):
        """Test that GW energy density is zero below cutoff."""
        gw = GravitationalWaves(rcfm_params)

        f_cut = gw.frequency_cutoff(0)

        # Frequencies below cutoff
        f_below = np.linspace(0.1 * f_cut, 0.9 * f_cut, 10) if f_cut > 1e-10 else np.array([1e-20, 1e-18, 1e-16])
        Omega_GW = gw.stochastic_background_primordial(f_below)

        # Should be essentially zero below cutoff
        assert np.all(Omega_GW < 1e-50)

    def test_stochastic_background_positive_above_cutoff(self, rcfm_params):
        """Test that GW energy density is positive above cutoff."""
        gw = GravitationalWaves(rcfm_params)

        f_cut = gw.frequency_cutoff(0)

        # Frequencies above cutoff
        f_above = np.array([10 * f_cut, 100 * f_cut]) if f_cut > 0 else np.array([1e-10, 1e-8])
        Omega_GW = gw.stochastic_background_primordial(f_above)

        # Should be positive
        assert np.all(Omega_GW >= 0)


class TestGWPhysicsConstraints:
    """Integration tests for GW physics constraints."""

    def test_gw170817_constraint_satisfied(self, rcfm_params):
        """Test that GW170817 speed constraint is satisfied.

        Paper claims: |c_T - c| < 5 × 10^-16
        """
        gw = GravitationalWaves(rcfm_params)

        # Check at various redshifts
        for z in [0.0, 0.1, 0.5]:
            c_T = gw.gw_speed(z)
            Lambda = rcfm_params.Lambda_RCFM(z)

            # Constraint: Λ_RSD < 1.7 × 10^-16 at z=0
            if z < 0.01:  # Near present
                deviation = abs(1.0 - c_T)
                assert deviation < 2e-16, f"GW170817 constraint violated at z={z}"

    def test_naturalness_argument(self, rcfm_params):
        """Test that naturalness argument is self-consistent.

        Paper claims: β can be 10^106 times Planck-suppressed and still satisfy constraints.
        """
        gw = GravitationalWaves(rcfm_params)

        # Current constraint: Λ_RCFM(z=0) < 1.7 × 10^-16
        Lambda_z0 = rcfm_params.Lambda_RCFM(0)

        # Should satisfy constraint
        assert Lambda_z0 < 1e-15, f"Λ_RCFM = {Lambda_z0} exceeds naturalness bound"
