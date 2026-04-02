#!/usr/bin/env python3
"""
Test suite for rcfm.core.metric module.

Tests metric tensor, scale factor, Hubble parameter, and curvature calculations.
Includes verification of BUG-FIX-4 and BUG-FIX-5.

Author: MiniMax Agent
Version: 1.3 (post bug fixes)
"""

import pytest
import numpy as np
from rcfm.core.metric import MetricRCFM, CurvatureCalculations
from rcfm.core.constants import get_rcfm_defaults


class TestMetricRCFM:
    """Tests for MetricRCFM class."""

    def test_initialization(self, rcfm_params):
        """Test metric initialization."""
        metric = MetricRCFM(rcfm_params)
        assert metric.params is not None
        assert metric.a0 == 1.0  # Present-day scale factor normalized to 1
        assert metric.r0 == rcfm_params.Rmax

    def test_scale_factor_at_present(self, rcfm_params):
        """Test scale factor at present day (r = Rmax)."""
        metric = MetricRCFM(rcfm_params)
        a = metric.scale_factor(rcfm_params.Rmax)
        assert np.isclose(a, 1.0, rtol=1e-10)

    def test_scale_factor_proportional_to_r_squared(self, rcfm_params):
        """Test that a(r) ∝ r² as per paper."""
        metric = MetricRCFM(rcfm_params)
        r0 = rcfm_params.Rmax

        # Check at two different r values
        r1 = 0.1 * r0
        r2 = 0.2 * r0

        a1 = metric.scale_factor(r1)
        a2 = metric.scale_factor(r2)

        # a1/a2 should equal (r1/r2)²
        expected_ratio = (r1/r2)**2
        actual_ratio = a1/a2
        assert np.isclose(actual_ratio, expected_ratio, rtol=1e-10)

    def test_scale_factor_analytic_derivative(self, rcfm_params):
        """Test analytical derivative da/dr matches analytical formula.

        For a(r) = a0 * (r/r0)²: da/dr = 2 * a0 * r / r0²
        """
        metric = MetricRCFM(rcfm_params)
        r = 0.5 * rcfm_params.Rmax

        # Get analytical derivative from method
        da_dr_analytic = metric.scale_factor_derivative(r)

        # Expected: da/dr = 2 * a0 * r / r0²
        expected = 2 * metric.a0 * r / (metric.r0**2)

        assert np.isclose(da_dr_analytic, expected, rtol=1e-10)

    def test_scale_factor_no_division_by_zero(self, rcfm_params):
        """BUG-FIX-4: Test that scale factor doesn't have singularity at r → 0.

        The method should use R_MIN to regularize.
        """
        metric = MetricRCFM(rcfm_params)

        # Should not raise an error or return NaN
        a_at_zero = metric.scale_factor(0)
        assert np.isfinite(a_at_zero)
        assert a_at_zero >= 0

        # Should not raise an error at very small r
        a_at_tiny = metric.scale_factor(1e-30)
        assert np.isfinite(a_at_tiny)
        assert a_at_tiny >= 0

    def test_hubble_parameter_finite_at_zero(self, rcfm_params):
        """BUG-FIX-5: Test that Hubble parameter is finite at r → 0.

        The method should use analytical derivative instead of
        numerical derivative that fails at r → 0.
        """
        metric = MetricRCFM(rcfm_params)

        # Should not raise an error or return NaN/Inf
        H_at_zero = metric.hubble_parameter(0)
        assert np.isfinite(H_at_zero)

        # Should not raise an error at very small r
        H_at_tiny = metric.hubble_parameter(1e-30)
        assert np.isfinite(H_at_tiny)

    def test_hubble_parameter_consistency(self, rcfm_params):
        """Test H(r) = (1/a) * da/dr with analytical formulas."""
        metric = MetricRCFM(rcfm_params)
        r = 0.5 * rcfm_params.Rmax

        a = metric.scale_factor(r)
        da_dr = metric.scale_factor_derivative(r)

        # Expected: H = da/dr / a
        expected_H = da_dr / a

        # From hubble_parameter method
        actual_H = metric.hubble_parameter(r)

        assert np.isclose(actual_H, expected_H, rtol=1e-10)

    def test_lapse_function_is_one(self, rcfm_params):
        """Test that lapse function is 1 (synchronous gauge)."""
        metric = MetricRCFM(rcfm_params)
        r = 0.5 * rcfm_params.Rmax

        A = metric.lapse_function(r)
        assert np.isclose(A, 1.0, rtol=1e-10)

    def test_metric_4d_signature(self, rcfm_params):
        """Test that 4D metric has correct signature (-,+,+,+)."""
        metric = MetricRCFM(rcfm_params)

        # At a test point
        r = 0.5 * rcfm_params.Rmax
        chi = np.pi / 4
        theta = np.pi / 3
        phi = np.pi / 6

        g = metric.metric_4D(r, chi, theta, phi)

        # Check signature: g_00 should be negative
        assert g[0, 0] < 0

        # Spatial components should be positive
        assert g[1, 1] > 0
        assert g[2, 2] > 0
        assert g[3, 3] > 0

    def test_inverse_metric_components(self, rcfm_params):
        """Test that g * g_inv = I."""
        metric = MetricRCFM(rcfm_params)

        r = 0.5 * rcfm_params.Rmax
        chi = np.pi / 4
        theta = np.pi / 3
        phi = np.pi / 6

        g = metric.metric_4D(r, chi, theta, phi)
        g_inv = metric.inverse_metric(r, chi, theta, phi)

        # Matrix product should be identity
        product = np.dot(g, g_inv)
        identity = np.eye(4)

        np.testing.assert_allclose(product, identity, rtol=1e-10)


class TestS3LaplacianEigenvalues:
    """Tests for S³ Laplacian eigenvalue calculations."""

    def test_eigenvalue_formula(self, rcfm_params):
        """Test eigenvalue formula: λ_n = -(n² - 1)."""
        metric = MetricRCFM(rcfm_params)

        for n in [1, 2, 5, 10, 100]:
            expected = -(n**2 - 1)
            actual = metric.S3_laplacian_eigenvalues(n)
            assert np.isclose(actual, expected, rtol=1e-10)

    def test_eigenvalue_negative(self, rcfm_params):
        """Test that eigenvalues are negative for n > 1."""
        metric = MetricRCFM(rcfm_params)

        for n in [2, 5, 10]:
            eigenvalue = metric.S3_laplacian_eigenvalues(n)
            assert eigenvalue < 0


class TestCurvatureCalculations:
    """Tests for CurvatureCalculations class."""

    def test_christoffel_symbols_shape(self, rcfm_params):
        """Test that Christoffel symbols have correct shape (4,4,4)."""
        metric = MetricRCFM(rcfm_params)
        curvature = CurvatureCalculations(metric)

        r = 0.5 * rcfm_params.Rmax
        chi = np.pi / 4
        theta = np.pi / 3
        phi = np.pi / 6

        Gamma = curvature.christoffel_symbols(r, chi, theta, phi)

        assert Gamma.shape == (4, 4, 4)

    def test_ricci_tensor_shape(self, rcfm_params):
        """Test that Ricci tensor has correct shape (4,4)."""
        metric = MetricRCFM(rcfm_params)
        curvature = CurvatureCalculations(metric)

        r = 0.5 * rcfm_params.Rmax
        chi = np.pi / 4
        theta = np.pi / 3
        phi = np.pi / 6

        R = curvature.ricci_tensor(r, chi, theta, phi)

        assert R.shape == (4, 4)

    def test_ricci_tensor_symmetric(self, rcfm_params):
        """Test that Ricci tensor is symmetric."""
        metric = MetricRCFM(rcfm_params)
        curvature = CurvatureCalculations(metric)

        r = 0.5 * rcfm_params.Rmax
        chi = np.pi / 4
        theta = np.pi / 3
        phi = np.pi / 6

        R = curvature.ricci_tensor(r, chi, theta, phi)

        # Check symmetry: R_μν = R_νμ
        np.testing.assert_allclose(R, R.T, rtol=1e-10)

    def test_ricci_scalar_positive(self, rcfm_params):
        """Test that Ricci scalar is positive for closed universe (S³)."""
        metric = MetricRCFM(rcfm_params)
        curvature = CurvatureCalculations(metric)

        # For closed FLRW with positive curvature, R > 0
        r = 0.5 * rcfm_params.Rmax
        R_scalar = curvature.ricci_scalar(r)

        assert R_scalar > 0

    def test_ricci_scalar_ricci_tensor_trace(self, rcfm_params):
        """Test that R = R^μ_μ (trace of Ricci tensor)."""
        metric = MetricRCFM(rcfm_params)
        curvature = CurvatureCalculations(metric)

        r = 0.5 * rcfm_params.Rmax
        chi = np.pi / 4
        theta = np.pi / 3
        phi = np.pi / 6

        R_scalar = curvature.ricci_scalar(r)
        R_tensor = curvature.ricci_tensor(r, chi, theta, phi)
        g_inv = metric.inverse_metric(r, chi, theta, phi)

        # R = g^{μν} R_{μν}
        R_trace = np.sum(g_inv * R_tensor)

        assert np.isclose(R_scalar, R_trace, rtol=1e-6)


class TestHubbleParameterEdgeCases:
    """Edge case tests for Hubble parameter calculation."""

    def test_hubble_parameter_at_present(self, rcfm_params):
        """Test Hubble parameter at present day."""
        metric = MetricRCFM(rcfm_params)
        H = metric.hubble_parameter(rcfm_params.Rmax)

        # Should be close to H0
        expected_H = 67.4  # km/s/Mpc
        H_m_s = H * 3.086e22 / 1000  # Convert to km/s/Mpc

        # Should be in reasonable range
        assert 50 < H_m_s < 100

    def test_hubble_parameter_early_universe(self, rcfm_params):
        """Test Hubble parameter in early universe (r << Rmax)."""
        metric = MetricRCFM(rcfm_params)

        r_early = 1e-5 * rcfm_params.Rmax
        H = metric.hubble_parameter(r_early)

        # H should be much larger in early universe
        H_present = metric.hubble_parameter(rcfm_params.Rmax)

        assert H > H_present
        assert np.isfinite(H)
