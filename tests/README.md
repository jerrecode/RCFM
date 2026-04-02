# RCFM Test Suite

This directory contains comprehensive tests for the RCFM (Radial-Cyclic Field Model) package.

## Running Tests

```bash
cd /workspace/files/RCFM
pytest tests/ -v
```

## Test Structure

- `conftest.py` - Pytest configuration and fixtures
- `test_constants.py` - Physical constants tests
- `test_metric.py` - Metric tensor tests
- `test_solver.py` - ODE solver tests
- `test_cmb.py` - CMB spectrum tests
- `test_gravitational_waves.py` - GW tests
- `test_integration.py` - End-to-end integration tests

## Bug Fix Verification

The test suite verifies all 7 critical bug fixes from the deep analysis report.
