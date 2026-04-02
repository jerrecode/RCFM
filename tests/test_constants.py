#!/usr/bin/env python3
"""
Test suite for rcfm.core.constants module.

Tests physical constants, parameter validation, and unit conversions.

Author: MiniMax Agent
Version: 1.3 (post bug fixes)
"""

import pytest
import numpy as np
from rcfm.core.constants import PhysicalConstants, RCFMParameters, get_rcfm_defaults


class TestPhysicalConstants:
    """Tests for PhysicalConstants class."""

    def test_constants_have_values(self, physical_constants):
        """Test that all constants have non-zero values."""
        assert physical_constants.c > 0
        assert physical_constants.G > 0
        assert physical_constants.hbar > 0
        assert physical_constants.planck_mass > 0

    def test_critical_density_calculated(self, physical_constants):
        """Test that critical density is calculated correctly."""
        # ρ_c = 3H²/(8πG)
        expected_rho_c = 3 * (physical_constants.H0_SI)**2 / (8 * np.pi * physical_constants.G)
        assert np.isclose(physical_constants.rho_crit_0, expected_rho_c, rtol=1e-10)

    def test_unit_conversions(self, physical_constants):
        """Test unit conversion factors."""
        # H0 in different units should be consistent
        H_km_s_Mpc = physical_constants.H0_SI * 3.086e22 / 1000  # 1/s to km/s/Mpc
        assert np.isclose(H_km_s_Mpc, physical_constants.H0, rtol=1e-10)

    def test_planck_mass(self, physical_constants):
        """Test Planck mass calculation."""
        # m_Pl = sqrt(hbar*c/G)
        expected_mp = np.sqrt(physical_constants.hbar * physical_constants.c / physical_constants.G)
        assert np.isclose(physical_constants.planck_mass, expected_mp, rtol=1e-10)


class TestRCFMParameters:
    """Tests for RCFMParameters class."""

    def test_default_initialization(self):
        """Test default parameter initialization."""
        params = RCFMParameters()
        assert params.beta > 0
        assert params.rho_B0 >= 0
        assert params.Gamma_drag > 0
        assert params.Rmax > 0

    def test_lambda_rcfm_at_z0(self, rcfm_params):
        """Test Λ_RCFM at z=0."""
        Lambda_z0 = rcfm_params.Lambda_RCFM(0)
        # At z=0, Λ_RCFM = 2 * β * ρ_B0 * Rmax³
        expected = 2 * rcfm_params.beta * rcfm_params.rho_B0 * (rcfm_params.Rmax)**3
        assert np.isclose(Lambda_z0, expected, rtol=1e-10)

    def test_lambda_rcfm_redshift_evolution(self, rcfm_params):
        """Test Λ_RCFM increases with redshift."""
        Lambda_z0 = rcfm_params.Lambda_RCFM(0)
        Lambda_z1 = rcfm_params.Lambda_RCFM(1)
        Lambda_z2 = rcfm_params.Lambda_RCFM(2)

        assert Lambda_z1 > Lambda_z0
        assert Lambda_z2 > Lambda_z1
        # Check scaling: Λ ∝ (1+z)³
        ratio = Lambda_z2 / Lambda_z0
        expected_ratio = 3**3  # (1+2)³ / (1+0)³
        assert np.isclose(ratio, expected_ratio, rtol=0.01)

    def test_lcdm_limit_zero_lambda(self, rcfm_params_lcdm_limit):
        """Test that Λ_RCFM → 0 when β=0 and ρ_B0=0."""
        Lambda = rcfm_params_lcdm_limit.Lambda_RCFM(0)
        assert np.isclose(Lambda, 0, atol=1e-30)

    def test_parameter_validation_positive_beta(self):
        """Test that negative β raises error."""
        with pytest.raises(ValueError):
            params = RCFMParameters(beta=-1.0)

    def test_parameter_validation_positive_rho_B0(self):
        """Test that negative ρ_B0 raises error."""
        with pytest.raises(ValueError):
            params = RCFMParameters(rho_B0=-1.0)

    def test_parameter_validation_positive_gamma(self):
        """Test that negative Γ_drag raises error."""
        with pytest.raises(ValueError):
            params = RCFMParameters(Gamma_drag=-1.0)


class TestGetRcfmDefaults:
    """Tests for get_rcfm_defaults function."""

    def test_returns_rcfm_parameters(self):
        """Test that function returns RCFMParameters instance."""
        params = get_rcfm_defaults()
        assert isinstance(params, RCFMParameters)

    def test_values_in_expected_ranges(self):
        """Test that default values are in expected physical ranges."""
        params = get_rcfm_defaults()

        # β should be very small (dimensionless coupling)
        assert 0 < params.beta < 1e-10

        # ρ_B0 should be much less than critical density
        rho_c = PhysicalConstants().rho_crit_0
        assert 0 < params.rho_B0 < rho_c

        # Γ_drag should be positive
        assert params.Gamma_drag > 0

        # Rmax should be in reasonable range (Hubble radius scale)
        assert 1e25 < params.Rmax < 1e28  # meters
