#!/usr/bin/env python3
"""
Pytest Configuration and Fixtures for RCFM Test Suite

Author: MiniMax Agent
Version: 1.3 (post bug fixes)
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rcfm.core.constants import PhysicalConstants, RCFMParameters, get_rcfm_defaults


@pytest.fixture
def physical_constants():
    """Provide PhysicalConstants instance."""
    return PhysicalConstants()


@pytest.fixture
def rcfm_params():
    """Provide RCFMParameters with default values."""
    return get_rcfm_defaults()


@pytest.fixture
def rcfm_params_lcdm_limit():
    """Provide RCFMParameters with β=0 (ΛCDM limit)."""
    params = get_rcfm_defaults()
    params.beta = 0.0
    params.rho_B0 = 0.0
    return params


@pytest.fixture
def r_values():
    """Provide typical r values for testing."""
    r_max = get_rcfm_defaults().Rmax
    return {
        'near_singularity': 1e-10 * r_max,
        'early': 0.1 * r_max,
        'present': r_max
    }


@pytest.fixture
def z_values():
    """Provide typical redshift values for testing."""
    return [0.0, 0.5, 1.0, 2.0, 5.0, 1100.0]


class TestConfig:
    """Test configuration constants."""
    # Numerical tolerances
    RTOL = 1e-5  # Relative tolerance for comparisons
    ATOL = 1e-12  # Absolute tolerance
    ENERGY_TOL = 1e-6  # Energy conservation tolerance

    # Physical tolerances
    GW_SPEED_TOL = 1e-15  # GW speed deviation from c
    HUBBLE_TOL = 0.1  # km/s/Mpc for H0 comparison


# Register custom markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "physics: marks tests as physics validation tests"
    )
