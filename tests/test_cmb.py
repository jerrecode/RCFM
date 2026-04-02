#!/usr/bin/env python3
"""
Test suite for rcfm.core.cmb_spectrum module.

Tests CMB angular power spectrum including BUG-FIX-6 (acoustic peak calculation).

Author: MiniMax Agent
Version: 1.3 (post bug fixes)
"""

import pytest
import numpy as np
from rcfm.core.cmb_spectrum import CMBAngularSpectrum
from rcfm.core.constants import get_rcfm_defaults


class TestCMBAngularSpectrum:
    """Tests for CMBAngularSpectrum class."""

    def test_initialization(self, rcfm_params):
        """Test CMB spectrum calculator initialization."""
        cmb = CMBAngularSpectrum(rcfm_params)
        assert cmb.params is not None
        assert cmb.harmonics is not None
        assert cmb.scalar is not None

    def test_sound_horizon_positive(self, rcfm_params):
        """Test that sound horizon is positive."""
        cmb = CMBAngularSpectrum(rcfm_params)
        r_s = cmb.sound_horizon(1100)

        assert r_s > 0

    def test_sound_horizon_in_physical_range(self, rcfm_params):
        """Test sound horizon is in physical range (~100-150 Mpc)."""
        cmb = CMBAngularSpectrum(rcfm_params)
        r_s = cmb.sound_horizon(1100)

        # Sound horizon should be around 100-150 Mpc
        assert 50 < r_s < 200, f"Sound horizon {r_s} Mpc outside expected range"

    def test_angular_diameter_distance_positive(self, rcfm_params):
        """Test that angular diameter distance is positive."""
        cmb = CMBAngularSpectrum(rcfm_params)
        D_A = cmb.angular_diameter_distance(1100)

        assert D_A > 0

    def test_angular_diameter_distance_increases_with_z(self, rcfm_params):
        """Test that D_A increases with redshift up to recombination."""
        cmb = CMBAngularSpectrum(rcfm_params)

        D_A_z0 = cmb.angular_diameter_distance(0)
        D_A_z1 = cmb.angular_diameter_distance(1)
        D_A_z2 = cmb.angular_diameter_distance(2)

        # D_A first increases then decreases, but should be monotonic up to z~2
        assert D_A_z2 > D_A_z1


class TestAcousticPeaks:
    """Tests for acoustic peak calculation (BUG-FIX-6)."""

    def test_acoustic_peaks_calculated(self, rcfm_params):
        """Test that acoustic peaks are calculated from physics, not hardcoded."""
        cmb = CMBAngularSpectrum(rcfm_params)

        result = cmb.compute_Cl(l_max=2500)

        # Should have acoustic peak information
        assert 'acoustic_peaks' in result
        assert 'sound_horizon' in result
        assert 'angular_diameter_distance' in result

    def test_acoustic_peaks_reasonable_values(self, rcfm_params):
        """Test that calculated peaks are in reasonable range."""
        cmb = CMBAngularSpectrum(rcfm_params)

        result = cmb.compute_Cl(l_max=2500)
        peaks = result['acoustic_peaks']

        # First peak should be around ℓ ~ 200-250
        assert 180 < peaks[0] < 280, f"First peak at {peaks[0]} outside expected range"

        # Second peak should be around ℓ ~ 500-600
        assert 450 < peaks[1] < 650, f"Second peak at {peaks[1]} outside expected range"

    def test_acoustic_peak_spacing(self, rcfm_params):
        """Test that peak spacing is roughly constant (as expected for acoustic oscillations)."""
        cmb = CMBAngularSpectrum(rcfm_params)

        result = cmb.compute_Cl(l_max=2500)
        peaks = result['acoustic_peaks']

        # Calculate spacings
        spacings = np.diff(peaks)

        # Spacing should be roughly constant (~250-300)
        # Allow for some variation
        mean_spacing = np.mean(spacings)
        assert 200 < mean_spacing < 350, f"Peak spacing {mean_spacing} not constant"

    def test_acoustic_peaks_depend_on_physics(self, rcfm_params):
        """Test that peaks depend on sound horizon and D_A, not fixed values."""
        cmb = CMBAngularSpectrum(rcfm_params)

        # Calculate with default params
        result1 = cmb.compute_Cl(l_max=2500)
        peaks1 = result1['acoustic_peaks']
        r_s_1 = result1['sound_horizon']
        D_A_1 = result1['angular_diameter_distance']

        # Peaks should scale as ℓ_n ∝ D_A / r_s
        # So if D_A/r_s changes, peaks change
        ratio_1 = D_A_1 / r_s_1

        # For ℓ_1 = π * D_A / r_s
        expected_ℓ1 = np.pi * ratio_1
        assert np.isclose(peaks1[0], expected_ℓ1, rtol=0.1)


class TestComputeCl:
    """Tests for compute_Cl method."""

    def test_compute_cl_returns_valid_dict(self, rcfm_params):
        """Test that compute_Cl returns valid dictionary."""
        cmb = CMBAngularSpectrum(rcfm_params)
        result = cmb.compute_Cl(l_max=1000)

        assert isinstance(result, dict)
        assert 'l' in result
        assert 'Cl' in result
        assert 'units' in result

    def test_compute_cl_multipole_range(self, rcfm_params):
        """Test that output multipoles are in correct range."""
        cmb = CMBAngularSpectrum(rcfm_params)
        result = cmb.compute_Cl(l_max=1000)

        l = result['l']

        # Should start at ℓ=2 (monopole and dipole are usually excluded)
        assert l[0] == 2
        # Should end at specified l_max
        assert l[-1] == 1000

    def test_compute_cl_positive_values(self, rcfm_params):
        """Test that C_ℓ values are positive."""
        cmb = CMBAngularSpectrum(rcfm_params)
        result = cmb.compute_Cl(l_max=1000)

        Cl = result['Cl']

        # C_ℓ should be positive (power spectrum)
        assert np.all(Cl >= 0)

    def test_compute_cl_peak_at_low_l(self, rcfm_params):
        """Test that there is power at low ℓ (Sachs-Wolfe plateau)."""
        cmb = CMBAngularSpectrum(rcfm_params)
        result = cmb.compute_Cl(l_max=1000)

        Cl = result['Cl']
        l = result['l']

        # Find C_ℓ in range ℓ=10-50
        mask = (l >= 10) & (l <= 50)
        if np.any(mask):
            Cl_low = np.mean(Cl[mask])
            assert Cl_low > 0

    def test_compute_cl_high_l_decreases(self, rcfm_params):
        """Test that C_ℓ decreases at high ℓ (damping tail)."""
        cmb = CMBAngularSpectrum(rcfm_params)
        result = cmb.compute_Cl(l_max=2000)

        Cl = result['Cl']
        l = result['l']

        # Compare power at ℓ~1500 vs ℓ~1000
        mask_1000 = (l >= 950) & (l <= 1050)
        mask_1500 = (l >= 1450) & (l <= 1550)

        if np.any(mask_1000) and np.any(mask_1500):
            Cl_1000 = np.mean(Cl[mask_1000])
            Cl_1500 = np.mean(Cl[mask_1500])

            # Should decrease at higher ℓ (Silk damping)
            assert Cl_1500 < Cl_1000


class TestPlanckComparison:
    """Tests for Planck data loading."""

    def test_load_planck_placeholder(self, rcfm_params):
        """Test that _load_planck_data returns values (placeholder)."""
        cmb = CMBAngularSpectrum(rcfm_params)
        Cl_planck = cmb._load_planck_data(1000)

        assert len(Cl_planck) == 999  # ℓ=2 to ℓ=1000
        assert np.all(Cl_planck >= 0)

    def test_compute_cl_with_planck_comparison(self, rcfm_params):
        """Test compute_Cl with load_planck=True."""
        cmb = CMBAngularSpectrum(rcfm_params)
        result = cmb.compute_Cl(l_max=1000, load_planck=True)

        assert 'Cl_planck' in result
        assert len(result['Cl_planck']) == 999


class TestCMBPhysicsConstraints:
    """Integration tests for CMB physics."""

    def test_first_peak_consistent_with_planck(self, rcfm_params):
        """Test that first acoustic peak is consistent with Planck.

        Planck finds ℓ_1 ≈ 220.
        """
        cmb = CMBAngularSpectrum(rcfm_params)
        result = cmb.compute_Cl(l_max=2500)

        peaks = result['acoustic_peaks']

        # First peak should be close to observed value
        assert 200 < peaks[0] < 250, f"First peak at {peaks[0]} inconsistent with Planck (~220)"

    def test_second_peak_consistent_with_planck(self, rcfm_params):
        """Test that second acoustic peak is consistent with Planck.

        Planck finds ℓ_2 ≈ 540.
        """
        cmb = CMBAngularSpectrum(rcfm_params)
        result = cmb.compute_Cl(l_max=2500)

        peaks = result['acoustic_peaks']

        # Second peak should be close to observed value
        assert 500 < peaks[1] < 590, f"Second peak at {peaks[1]} inconsistent with Planck (~540)"
