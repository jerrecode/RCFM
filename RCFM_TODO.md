# RCFM Implementation TODO List
## Radial-Cyclic Field Model - Complete Task Checklist

**Author:** MiniMax Agent
**Generated:** 2026-03-28
**Based On:** RCFM_Deep_Analysis_Report.md + Unified Implementation Plan
**Goal:** Publication-ready scientific codebase

---

## Overview

This TODO consolidates all implementation tasks derived from the deep analysis into a single actionable checklist. The RCFM project is currently at **prototype stage (~40% complete)** with critical bugs that must be fixed before publication.

---

## PHASE 1: Critical Bug Fixes

### [BUG-FIX-1] Raychaudhuri Equation Sign Error ⚠️ CRITICAL
- [ ] **File:** `src/rcfm/core/solver.py`
- [ ] **Line:** 72-73
- [ ] **Issue:** Sign of pressure term reversed in Raychaudhuri equation
- [ ] **Current (WRONG):**
  ```python
  dH_dr = sqrt_A * (4 * np.pi * self.G / 3 * (rho_eff + 3 * p_eff) - H**2)
  ```
- [ ] **Fixed (CORRECT):**
  ```python
  dH_dr = sqrt_A * (-(4 * np.pi * self.G / 3) * (rho_eff + 3 * p_eff) - H**2)
  ```
- [ ] **Verification:** Compare against ΛCDM limit using CAMB/CLASS
- [ ] **Impact:** Wrong acceleration/deceleration evolution

---

### [BUG-FIX-2] EMT Effective Pressure Formula Mismatch
- [ ] **File:** `src/rcfm/core/fluids.py`
- [ ] **Line:** 140
- [ ] **Issue:** Uses `Lambda_RCFM(0)` instead of `H` variable, factor of 3 mismatch
- [ ] **Current (WRONG):**
  ```python
  p_eff = p_A - beta * self.params.Gamma_drag * rho_A * rho_B * (rho_A - rho_B) / (3 * self.params.Lambda_RCFM(0))
  ```
- [ ] **Fix:** Replace `Lambda_RCFM(0)` with `H` and verify factor of 3
- [ ] **Verification:** Test energy conservation ∇_μ T^{μν} = 0

---

### [BUG-FIX-3] G_eff Formula Incorrect in GW Mass Term
- [ ] **File:** `src/rcfm/core/gravitational_waves.py`
- [ ] **Line:** 80
- [ ] **Issue:** Wrong formula for effective gravitational constant
- [ ] **Current (WRONG):**
  ```python
  G_eff = self.G * (1 + self.params.Lambda_RCFM(z))
  ```
- [ ] **Fixed (CORRECT - per paper eq. 523-525):**
  ```python
  a = 1.0 / (1 + z)
  G_eff = self.G * (1 + self.params.beta * self.params.rho_B0 * (self.params.Rmax / a)**3)
  ```
- [ ] **Impact:** Wrong GW cutoff frequency

---

### [BUG-FIX-4] Scale Factor Division by Zero
- [ ] **File:** `src/rcfm/core/metric.py`
- [ ] **Lines:** 54-56
- [ ] **Issue:** `a(r) = a0 * (r/r0)^2 → 0` at r → 0 causes division by zero in H = a'/a
- [ ] **Fix Options:**
  - [ ] Option A: Enforce `r_min > 0` in solver initialization
  - [ ] Option B: Regularize scale factor near singularity: `a = max(a0 * (r/r0)**2, eps)`
  - [ ] Option C: Use analytical formula for H near r=0
- [ ] **Impact:** NaN/Inf at r → 0

---

### [BUG-FIX-5] Hubble Derivative Logic Error
- [ ] **File:** `src/rcfm/core/metric.py`
- [ ] **Line:** 162
- [ ] **Issue:** `else: if r > 0` is unreachable logic error
- [ ] **Current (WRONG):**
  ```python
  else:
      da_dr = 2 * a / r if r > 0 else 0  # Logic error!
  ```
- [ ] **Fixed (ANALYTICAL):**
  ```python
  # For a(r) = a0 * (r/r0)^2, analytical derivative is:
  # da/dr = 2 * a0 * r / r0^2
  r0 = self.params.Rmax
  a0 = 1.0  # or self.params.a0
  da_dr = 2 * a0 * r / (r0**2)
  ```
- [ ] **Impact:** Unreachable else branch, potential division by zero

---

### [BUG-FIX-6] Hardcoded CMB Peak Positions
- [ ] **File:** `src/rcfm/core/cmb_spectrum.py`
- [ ] **Line:** 191
- [ ] **Issue:** Peak positions [220, 540, 810, ...] are ΛCDM values hardcoded
- [ ] **Fix Required:**
  - [ ] Calculate acoustic peaks from sound horizon `rs` and angular diameter distance `DA`
  - [ ] Implement acoustic oscillation integral
  - [ ] Formula: `ℓ_n ≈ n * π * DA / rs`
  - [ ] Remove hardcoded peak array
- [ ] **Impact:** CMB predictions not derived from RCFM physics

---

### [BUG-FIX-7] No Solution Success Checks
- [ ] **File:** `src/rcfm/core/solver.py`
- [ ] **Lines:** 134-143
- [ ] **Issue:** `solution.success` returned but never checked by calling code
- [ ] **Fix Required:**
  - [ ] Add validation in `BackgroundSolver.solve()` method
  - [ ] Add validation in all calling code (CLI, likelihood, etc.)
  - [ ] Raise `RuntimeError` or return error code on failure
  - [ ] Add test for failed integration scenario
- [ ] **Impact:** Silent failures propagate through pipeline

---

## PHASE 2: Missing Implementations

### [IMPL-1] Christoffel Symbols (Placeholder → Real)
- [ ] **File:** `src/rcfm/core/metric.py`
- [ ] **Line:** 234
- [ ] **Status:** Returns zeros (placeholder)
- [ ] **Required:**
  - [ ] Calculate actual Christoffel symbols for 4-ball metric
  - [ ] Formula for metric `ds² = -A(r)dr² + a(r)²dΩ₃²`
  - [ ] Implement: Γ^μ_νρ = (1/2) g^μσ (∂_ν g_σρ + ∂_ρ g_σν - ∂_σ g_νρ)
  - [ ] Verify: Newtonian limit yields ∇²Φ = 4πGρ
- [ ] **Tests:** `test_christoffel_newtonian_limit()`

---

### [IMPL-2] Tight-Coupling Boltzmann Hierarchy
- [ ] **Status:** NOT IMPLEMENTED
- [ ] **Required for:** CMB photon-baryon fluid evolution, Silk damping
- [ ] **Components:**
  - [ ] Photon distribution function
  - [ ] Boltzmann equation for photons
  - [ ] Thomson scattering term
  - [ ] Baryon momentum equation
  - [ ] Tight-coupling expansion (ε << 1)
- [ ] **Tests:** `test_tight_coupling_approximation()`

---

### [IMPL-3] Actual Planck Data Loading
- [ ] **File:** `src/rcfm/core/cmb_spectrum.py`
- [ ] **Status:** Uses fake Gaussian peaks
- [ ] **Required:**
  - [ ] Parse `COM_PowerSpect_CMB-TT-full_R3.01.txt`
  - [ ] Parse `COM_PowerSpect_CMB-TE-full_R3.01.txt`
  - [ ] Parse `COM_PowerSpect_CMB-EE-full_R3.01.txt`
  - [ ] Load covariance matrices
  - [ ] Implement proper likelihood with cosmic variance
  - [ ] Replace `_load_planck_data()` placeholder
- [ ] **Data Files:** Already in `Data/raw/planck_2018/`

---

### [IMPL-4] Primordial Spectrum from Singularity
- [ ] **File:** `src/rcfm/core/perturbations.py`
- [ ] **Status:** `n_s = 0.965` hardcoded
- [ ] **Required:**
  - [ ] Derive n_s from a ∝ r² boundary condition via Bogoliubov transformation
  - [ ] Implement quantum mode function near singularity
  - [ ] Match to Bunch-Davies vacuum
  - [ ] Extract power spectrum P(k) ∝ k^(n_s - 1)
  - [ ] Verify n_s ≈ 0.965 emerges naturally
- [ ] **Paper Reference:** Section 10, Lines 699-706

---

### [IMPL-5] Ginzburg-Landau Condensate
- [ ] **Status:** NOT IMPLEMENTED
- [ ] **Required for:** Critical density ρc determination
- [ ] **Components:**
  - [ ] Define order parameter φ (gauge condensate)
  - [ ] Free energy functional F[φ]
  - [ ] Phase transition at ρ = ρc where ⟨φ⟩ → 0
  - [ ] Calculate ρc from coupling constants
  - [ ] Integrate with fluids.py for Phase B identification
- [ ] **Paper Reference:** Section 4.6, Lines 527-546
- [ ] **Tests:** `test_phase_transition_critical_density()`

---

### [IMPL-6] Bogoliubov Transformation (Entropy)
- [ ] **Status:** NOT IMPLEMENTED
- [ ] **Required for:** Thermodynamic entropy reset verification
- [ ] **Components:**
  - [ ] Bogoliubov coefficients α_k, β_k
  - [ ] S-matrix for singularity passage
  - [ ] Entropy calculation: S_out = Σ_k [...]
  - [ ] Verify entropy reset to zero
  - [ ] Connection to Page's theorem analogy
- [ ] **Paper Reference:** Section 13, Lines 774-794
- [ ] **Tests:** `test_entropy_reset()`

---

### [IMPL-7] DESI/Pantheon+ Data Loading
- [ ] **Status:** NOT USED (data downloaded but unused)
- [ ] **Required:**
  - [ ] Parse DESI BAO chains from `Data/raw/desi_2024/`
  - [ ] Parse Pantheon+ CSV files from `Data/raw/pantheon_plus/`
  - [ ] Parse GW170817 posterior samples
  - [ ] Implement proper likelihood comparison
- [ ] **Current Issue:** Likelihood code just uses penalty on Λ

---

### [IMPL-8] Production MCMC Sampler
- [ ] **File:** `src/rcfm/core/likelihood.py`
- [ ] **Status:** Toy Metropolis-Hastings implementation
- [ ] **Required:**
  - [ ] Replace with emcee or PyMC3
  - [ ] Implement adaptive proposal distribution
  - [ ] Add convergence diagnostics (Gelman-Rubin, R̂)
  - [ ] Calculate effective sample size (ESS)
  - [ ] Proper posterior output packaging
- [ ] **Parameters to sample:** β, ρ_B0, H0 (if not fixed)

---

### [IMPL-9] Comprehensive Test Suite
- [ ] **Status:** ZERO unit tests
- [ ] **Create `tests/` directory structure:**
  ```
  tests/
  ├── __init__.py
  ├── conftest.py              # pytest fixtures
  ├── test_constants.py
  ├── test_metric.py
  ├── test_fluids.py
  ├── test_solver.py
  ├── test_perturbations.py
  ├── test_cmb.py
  ├── test_matter_power.py
  ├── test_gravitational_waves.py
  ├── test_likelihood.py
  └── test_integration.py
  ```
- [ ] **Critical tests to implement:**
  - [ ] `test_lcdm_limit()` - β=0 reproduces ΛCDM
  - [ ] `test_energy_conservation()` - ∇_μ T^{μν} = 0
  - [ ] `test_exact_solution()` - a ∝ r² satisfies ODEs
  - [ ] `test_newtonian_limit()` - ∇²Φ = 4πGρ
  - [ ] `test_convergence()` - solution converges with finer grid
  - [ ] `test_gw_speed()` - c_T = 1 constraint satisfied
  - [ ] `test_scale_factor_analytic()` - analytical H matches numerical

---

### [IMPL-10] Ricci Tensor Calculations
- [ ] **File:** `src/rcfm/core/metric.py`
- [ ] **Status:** Uses approximate derivatives
- [ ] **Required:**
  - [ ] Calculate accurate Riemann tensor components
  - [ ] Contract to Ricci tensor R_μν
  - [ ] Scalar curvature R
  - [ ] Verify Einstein equations: G_μν = 8πT_μν
- [ ] **Tests:** `test_einstein_equations()`

---

### [IMPL-11] Reionization Implementation
- [ ] **Status:** NOT IMPLEMENTED
- [ ] **Required for:** CMB optical depth τ
- [ ] **Components:**
  - [ ] Optical depth integral τ(z) = ∫ n_e σ_T dz
  - [ ] Thomson scattering coefficient
  - [ ] Reionization redshift z_re ~ 7-8
  - [ ] Impact on CMB E-mode polarization
- [ ] **Tests:** `test_reionization_optical_depth()`

---

### [IMPL-12] Silk Damping
- [ ] **Status:** NOT IMPLEMENTED
- [ ] **Required for:** CMB damping tail at high ℓ
- [ ] **Components:**
  - [ ] Photon diffusion length
  - [ ] Damping factor exp(-k²/k_D²)
  - [ ] Integration with CMB spectrum code
- [ ] **Tests:** `test_silk_damping()`

---

## PHASE 3: File-Level Tasks

### `src/rcfm/__init__.py`
- [ ] [INIT-1] Add package metadata (version, author, license)
- [ ] [INIT-2] Export public API (RCFMParameters, BackgroundSolver, etc.)

---

### `src/rcfm/cli.py`
- [ ] [CLI-1] Implement CLI entry point with argparse
- [ ] [CLI-2] Add model execution commands (run, plot, analyze)
- [ ] [CLI-3] Add validation and debugging commands

---

### `src/rcfm/core/constants.py`
- [ ] [CONST-1] Document all physical constants with references
- [ ] [CONST-2] Add unit conversion utilities (SI ↔ natural units)

---

### `src/rcfm/core/metric.py`
- [ ] [METRIC-1] Complete implementation (bug fixes + IMPL-1, IMPL-10)
- [ ] [METRIC-2] Add curvature tensor validation
- [ ] [IMPL-1] Christoffel symbols
- [ ] [IMPL-10] Ricci tensor

---

### `src/rcfm/core/fluids.py`
- [ ] [FLUID-1] Verify EMT implementation (BUG-FIX-2)
- [ ] [FLUID-2] Implement drag terms with disable flag
- [ ] [IMPL-5] Ginzburg-Landau condensate

---

### `src/rcfm/core/solver.py`
- [ ] [SOLVER-1] Fix solution success checks (BUG-FIX-7)
- [ ] [SOLVER-2] Add input validation
- [ ] [SOLVER-3] Improve error handling and reporting
- [ ] [BUG-FIX-1] Raychaudhuri sign error

---

### `src/rcfm/core/perturbations.py`
- [ ] [PERT-1] Implement Boltzmann hierarchy (IMPL-2)
- [ ] [PERT-2] Derive n_s from boundary condition (IMPL-4)
- [ ] [PERT-3] Add gauge diagnostics
- [ ] [IMPL-6] Bogoliubov transformation

---

### `src/rcfm/core/cmb_spectrum.py`
- [ ] [CMB-1] Implement CMB from first principles
- [ ] [CMB-2] Add acoustic peak calculation
- [ ] [BUG-FIX-6] Remove hardcoded peaks
- [ ] [IMPL-3] Load actual Planck data
- [ ] [IMPL-11] Reionization
- [ ] [IMPL-12] Silk damping

---

### `src/rcfm/core/matter_power.py`
- [ ] [MP-1] Implement with RCFM transfer function
- [ ] [MP-2] Add spectral residual reporting

---

### `src/rcfm/core/gravitational_waves.py`
- [ ] [GW-1] Full tensor sector implementation
- [ ] [GW-2] Stochastic background + cutoff diagnostics
- [ ] [BUG-FIX-3] Fix G_eff formula

---

### `src/rcfm/core/likelihood.py`
- [ ] [LIK-1] Strict likelihood interface
- [ ] [LIK-2] Add priors and nuisance handling
- [ ] [LIK-3] Posterior output packaging
- [ ] [IMPL-7] Load DESI/Pantheon+ data
- [ ] [IMPL-8] Production MCMC

---

### `src/rcfm/viz/plotting.py`
- [ ] [VIZ-1] Reproducible plotting layer
- [ ] [VIZ-2] Add provenance overlays
- [ ] [VIZ-3] Residual plot templates

---

## PHASE 4: Global Audits

### [G-1] Repository-wide Audit
- [ ] Inventory all `src/rcfm/*` files and import graph
- [ ] Record every public function, class, CLI entry point
- [ ] Identify dead code, duplicated math, untested branches
- [ ] Establish execution path: CLI → core → plotting → outputs
- [ ] Map all 7 bugs to specific files/lines

---

### [G-2] Scientific Assumptions Audit
- [ ] Identify every model-defining assumption
- [ ] Separate "derived from equations" vs "chosen modeling ansatz"
- [ ] Flag manuscript claims stronger than code evidence
- [ ] Map all 7 theoretical gaps to sections
- [ ] Map all 12 missing implementations

---

### [G-3] Units and Dimension Audit
- [ ] Verify dimensional consistency at solver boundary
- [ ] Establish units policy (internal natural units, SI at boundaries)
- [ ] Add explicit conversion helpers
- [ ] Flag unit-related bugs

---

### [G-4] Reproducibility Baseline
- [ ] Define clean-room rebuild procedure
- [ ] Fix random seeds
- [ ] Ensure outputs regenerable from source + data
- [ ] Create `requirements.txt` or `environment.yml`
- [ ] Add [IMPL-9] test suite

---

### [G-5] Scientific Baseline Comparisons
- [ ] Build comparison harnesses vs ΛCDM baseline (CAMB/CLASS)
- [ ] Define residual metrics and acceptance thresholds
- [ ] Store baseline outputs for regression comparison
- [ ] Verify BUG-FIX-6 produces correct CMB predictions

---

## PHASE 5: Documentation

### Paper Updates
- [ ] Update to v1.4 with numerical results
- [ ] Add figures: H(z), C_ℓ, P(k), posteriors
- [ ] Revise theoretical sections with complete derivations
- [ ] Add appendix with numerical methods
- [ ] Finalize bibliography

---

### Code Documentation
- [ ] Add mathematical equation references to docstrings
- [ ] Document all public APIs
- [ ] Add examples to docstrings
- [ ] Create API reference guide
- [ ] Document test coverage

---

### README Updates
- [ ] Add Python environment setup instructions
- [ ] Add example usage scripts
- [ ] Document expected outputs
- [ ] Add computational requirements
- [ ] Add contribution guidelines

---

## Acceptance Criteria

### For Publication Readiness
- [ ] All 7 bugs fixed
- [ ] All 7 theoretical gaps addressed
- [ ] All 12 missing implementations completed
- [ ] Test suite > 80% coverage on core modules
- [ ] Baseline comparison shows < 0.1% deviation from ΛCDM limit
- [ ] All figures reproducible
- [ ] Documentation complete
- [ ] Code versioned and archived

---

## Estimated Timeline

| Phase | Duration | Focus |
|-------|----------|-------|
| Phase 1 | 2-4 weeks | Bug fixes |
| Phase 2 | 4-8 weeks | Missing derivations |
| Phase 3 | 6-10 weeks | Numerical implementations |
| Phase 4 | 3-5 weeks | Data integration |
| Phase 5 | 4-6 weeks | Testing & validation |
| Phase 6 | 2-4 weeks | Parameter estimation |
| Phase 7 | 4-6 weeks | Publication prep |
| **Total** | **25-43 weeks** | **~6-10 months** |

---

## Risk Register

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Theoretical gaps prove insurmountable | High | Medium | Focus on verifiable ΛCDM predictions |
| Numerical instabilities near singularity | High | High | Regularization + adaptive step size |
| Data parsing complexity underestimated | Medium | Medium | Allocate buffer time |
| MCMC convergence issues | Medium | Low | Use established samplers (emcee) |
| Reviewer challenges model assumptions | High | High | Strengthen theoretical justification |

---

*Last Updated: 2026-03-28*
*Status: Phase 1 - Bug Fixing: 0/7 complete*
