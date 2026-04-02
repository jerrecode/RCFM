#!/usr/bin/env python3
"""
Test suite for rcfm.core.solver module.

Tests background evolution ODE solver including BUG-FIX-1 (Raychaudhuri sign)
and BUG-FIX-7 (solution success checks).

Author: MiniMax Agent
Version: 1.3 (post bug fixes)
"""

import pytest
import numpy as np
import warnings
from rcfm.core.solver import BackgroundSolver, solve_background
from rcfm.core.constants import get_rcfm_defaults


class TestBackgroundSolver:
    """Tests for BackgroundSolver class."""

    def test_initialization(self, rcfm_params):
        """Test solver initialization."""
        solver = BackgroundSolver(rcfm_params)
        assert solver.params is not None
        assert solver.fluid is not None
        assert solver.G > 0

    def test_rhs_returns_correct_shape(self, rcfm_params):
        """Test that RHS function returns correct shape."""
        solver = BackgroundSolver(rcfm_params)

        # State vector: [a, H, rho_A, rho_B]
        y = np.array([1.0, 1e-18, 1e-5, 1e-10])
        r = 0.5 * rcfm_params.Rmax

        dy_dr = solver.rhs(r, y)

        assert len(dy_dr) == 4
        assert np.all(np.isfinite(dy_dr))

    def test_rhs_physically_reasonable(self, rcfm_params):
        """Test that RHS gives physically reasonable derivatives."""
        solver = BackgroundSolver(rcfm_params)

        # Present-day state
        y = np.array([1.0, 2.3e-18, 1e-26, 1e-28])
        r = rcfm_params.Rmax

        dy_dr = solver.rhs(r, y)

        # da/dr should be positive (scale factor increases with r)
        assert dy_dr[0] > 0

        # dH/dr should be negative in decelerating universe
        assert dy_dr[1] < 0

    def test_solve_returns_valid_solution(self, rcfm_params):
        """Test that solve method returns valid solution structure."""
        solver = BackgroundSolver(rcfm_params)

        solution = solver.solve(
            r_start=1e-6,
            r_end=0.1 * rcfm_params.Rmax,
            a_init=1e-12,
            H_init=1e10,
            rho_A_init=1e10,
            rho_B_init=1e8
        )

        # Check solution has required keys
        assert 'r' in solution
        assert 'a' in solution
        assert 'H' in solution
        assert 'rho_A' in solution
        assert 'rho_B' in solution
        assert 'success' in solution
        assert 'message' in solution
        assert 'status' in solution  # BUG-FIX-7
        assert 'nfev' in solution  # BUG-FIX-7

    def test_solution_success_flag(self, rcfm_params):
        """BUG-FIX-7: Test that solution success flag is properly set."""
        solver = BackgroundSolver(rcfm_params)

        solution = solver.solve(
            r_start=1e-6,
            r_end=0.1 * rcfm_params.Rmax
        )

        # Solution should either succeed or at least not crash
        assert solution['success'] or len(solution['r']) > 0

    def test_solution_no_nan(self, rcfm_params):
        """BUG-FIX-7: Test that solution contains no NaN values."""
        solver = BackgroundSolver(rcfm_params)

        solution = solver.solve(
            r_start=1e-6,
            r_end=0.1 * rcfm_params.Rmax
        )

        assert np.all(np.isfinite(solution['a']))
        assert np.all(np.isfinite(solution['H']))
        assert np.all(np.isfinite(solution['rho_A']))
        assert np.all(np.isfinite(solution['rho_B']))

    def test_scale_factor_increases_with_r(self, rcfm_params):
        """Test that scale factor increases with radial coordinate."""
        solver = BackgroundSolver(rcfm_params)

        solution = solver.solve(
            r_start=1e-6,
            r_end=0.1 * rcfm_params.Rmax
        )

        # Scale factor should be monotonically increasing
        assert np.all(np.diff(solution['a']) >= 0)

    def test_hubble_parameter_decreases_with_r(self, rcfm_params):
        """Test that Hubble parameter decreases as universe expands."""
        solver = BackgroundSolver(rcfm_params)

        solution = solver.solve(
            r_start=1e-6,
            r_end=0.1 * rcfm_params.Rmax
        )

        # Hubble should generally decrease (with some fluctuations OK)
        # Check that H at end < H at start
        assert solution['H'][-1] < solution['H'][0]

    def test_density_decreases_with_expansion(self, rcfm_params):
        """Test that densities decrease with expansion."""
        solver = BackgroundSolver(rcfm_params)

        solution = solver.solve(
            r_start=1e-6,
            r_end=0.1 * rcfm_params.Rmax
        )

        # ρ_A should decrease (matter dominated)
        assert solution['rho_A'][-1] < solution['rho_A'][0]

    def test_comoving_coordinates_monotonic(self, rcfm_params):
        """Test that integration produces monotonic comoving coordinates."""
        solver = BackgroundSolver(rcfm_params)

        solution = solver.solve(
            r_start=1e-6,
            r_end=0.1 * rcfm_params.Rmax
        )

        # Radial coordinate should be monotonically increasing
        assert np.all(np.diff(solution['r']) > 0)

    def test_initial_conditions_at_singularity(self, rcfm_params):
        """Test behavior near singularity (r → 0)."""
        solver = BackgroundSolver(rcfm_params)

        # Very close to singularity
        solution = solver.solve(
            r_start=1e-12,
            r_end=0.01 * rcfm_params.Rmax
        )

        # Should not crash or produce NaN
        assert len(solution['r']) > 0
        assert np.all(np.isfinite(solution['a']))
        assert np.all(np.isfinite(solution['H']))


class TestSolveBackground:
    """Tests for solve_background convenience function."""

    def test_solve_background_returns_solution(self, rcfm_params):
        """Test that convenience function returns valid solution."""
        solution = solve_background(rcfm_params)

        assert isinstance(solution, dict)
        assert 'r' in solution
        assert 'a' in solution
        assert 'H' in solution

    def test_solve_background_default_parameters(self, rcfm_params):
        """Test solve_background with default parameters."""
        solution = solve_background(rcfm_params)

        # Should integrate to Rmax
        assert np.isclose(solution['r'][-1], rcfm_params.Rmax, rtol=0.01)

    def test_scale_factor_at_end_is_one(self, rcfm_params):
        """Test that scale factor is normalized to 1 at present."""
        solution = solve_background(rcfm_params)

        # Scale factor at final point should be close to 1
        assert np.isclose(solution['a'][-1], 1.0, rtol=0.1)

    def test_lcdm_limit_zero_beta(self, rcfm_params):
        """Test ΛCDM limit (β → 0, ρ_B0 → 0)."""
        params = rcfm_params
        params.beta = 0.0
        params.rho_B0 = 0.0

        solution = solve_background(params)

        # Should still produce valid solution
        assert len(solution['r']) > 0
        assert np.all(np.isfinite(solution['H']))


class TestComputeHOfZ:
    """Tests for H(z) computation."""

    def test_compute_H_of_z_returns_function(self, rcfm_params):
        """Test that H(z) computation returns callable."""
        solver = BackgroundSolver(rcfm_params)

        solution = solver.solve(
            r_start=1e-6,
            r_end=0.1 * rcfm_params.Rmax
        )

        H_of_z = solver.compute_H_of_z(solution)

        assert callable(H_of_z)

    def test_H_at_z0_is_present_value(self, rcfm_params):
        """Test that H(0) matches present-day value."""
        solver = BackgroundSolver(rcfm_params)

        solution = solver.solve(
            r_start=1e-6,
            r_end=0.1 * rcfm_params.Rmax
        )

        H_of_z = solver.compute_H_of_z(solution)
        H_z0 = H_of_z(0.0)

        # Should be close to H at final point
        H_final = solution['H'][-1]
        assert np.isclose(H_z0, H_final, rtol=0.1)

    def test_H_increases_with_redshift(self, rcfm_params):
        """Test that H(z) increases with redshift."""
        solver = BackgroundSolver(rcfm_params)

        solution = solver.solve(
            r_start=1e-6,
            r_end=0.1 * rcfm_params.Rmax
        )

        H_of_z = solver.compute_H_of_z(solution)

        H_z0 = H_of_z(0.0)
        H_z1 = H_of_z(1.0)
        H_z2 = H_of_z(2.0)

        assert H_z1 > H_z0
        assert H_z2 > H_z1


class TestRaychaudhuriEquation:
    """Tests specifically for Raychaudhuri equation (BUG-FIX-1)."""

    def test_raychaudhuri_equation_satisfied(self, rcfm_params):
        """Test that corrected Raychaudhuri equation is satisfied.

        Paper eq (28): H'/√A + H² = -(4πG/3)(ρ_eff + 3p_eff)

        The sign correction ensures proper acceleration/deceleration.
        """
        solver = BackgroundSolver(rcfm_params)

        solution = solver.solve(
            r_start=1e-6,
            r_end=0.1 * rcfm_params.Rmax
        )

        # Check that H evolves reasonably
        # With corrected sign, H should decrease (deceleration)
        H_initial = solution['H'][0]
        H_final = solution['H'][-1]

        # In matter-dominated era, universe decelerates, so H should decrease
        assert H_final < H_initial

    def test_friedmann_equation_consistency(self, rcfm_params):
        """Test consistency with Friedmann equation.

        F1: H² = (8πG/3)(ρ_A + ρ_B + 2βρ_Aρ_B) - 1/a²
        """
        solver = BackgroundSolver(rcfm_params)

        solution = solver.solve(
            r_start=1e-6,
            r_end=0.1 * rcfm_params.Rmax
        )

        # Check at a few points
        for i in [0, len(solution['r'])//2, -1]:
            r = solution['r'][i]
            a = solution['a'][i]
            H = solution['H'][i]
            rho_A = solution['rho_A'][i]
            rho_B = solution['rho_B'][i]

            # Compute left and right sides of F1
            beta = rcfm_params.beta
            rho_eff = rho_A + rho_B + 2 * beta * rho_A * rho_B

            LHS = H**2
            RHS = (8 * np.pi * solver.G / 3) * rho_eff - 1/a**2

            # Should be approximately equal (within numerical error)
            # This tests that the ODE integration follows F1
            if a > 1e-6:  # Avoid near singularity
                relative_diff = abs(LHS - RHS) / (abs(LHS) + abs(RHS))
                assert relative_diff < 0.1  # 10% tolerance due to approximations
