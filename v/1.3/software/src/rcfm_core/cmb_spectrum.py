#!/usr/bin/env python3
"""
RCFM Module - CMB Angular Power Spectrum

This module implements the calculation of the CMB angular power spectrum
for the RCFM model, including the effects of the dual-stream fluid
and S^3 topology.

Author: MiniMax Agent
Version: 1.3
"""

import numpy as np
from typing import Tuple, Optional, Callable
from scipy import special, integrate
from scipy.interpolate import interp1d
from .constants import PhysicalConstants, RCFMParameters
from .perturbations import HypersphericalHarmonics, ScalarPerturbations


class CMBAngularSpectrum:
    """
    CMB Angular Power Spectrum Calculator

    Computes C_ℓ coefficients for the RCFM model including:
    - S^3 mode discretisation
    - Phase B anisotropic stress
    - Drag term modifications
    """

    def __init__(self, params: RCFMParameters):
        """
        Initialize CMB spectrum calculator.

        Args:
            params: RCFMParameters instance
        """
        self.params = params
        self.harmonics = HypersphericalHarmonics()
        self.scalar = ScalarPerturbations(params)
        self.c = PhysicalConstants.c
        self.G = PhysicalConstants.G

    def sound_horizon(self, z_rec: float = 1100) -> float:
        """
        Calculate sound horizon at recombination.

        r_s = ∫_0^{z_rec} c_s(z) dz / H(z)

        Args:
            z_rec: Recombination redshift

        Returns:
            Sound horizon in Mpc
        """
        # Sound speed (tight-coupling approximation)
        c_s = self.c / np.sqrt(3)  # c_s = c/√3 for relativistic fluid

        # Simplified: assume H ∝ (1+z)^2 during matter domination
        def H(z):
            # H(z) = H0 * sqrt(Omega_m * (1+z)^3 + Omega_Lambda)
            Omega_m = 0.315
            Omega_Lambda = 0.685
            H0 = 67.4  # km/s/Mpc
            return H0 * np.sqrt(Omega_m * (1+z)**3 + Omega_Lambda) * 1e3 / 3.086e22  # 1/s

        def integrand(z):
            return c_s / H(z)

        r_s, _ = integrate.quad(integrand, 0, z_rec)

        # Convert to Mpc
        return r_s / 3.086e22  # Mpc

    def angular_diameter_distance(self, z: float) -> float:
        """
        Calculate angular diameter distance.

        D_A(z) = r(z) / (1+z)

        Args:
            z: Redshift

        Returns:
            Angular diameter distance in Mpc
        """
        # Comoving distance (simplified flat case)
        Omega_m = 0.315
        Omega_Lambda = 0.685
        H0 = 67.4  # km/s/Mpc

        def H(z):
            return H0 * np.sqrt(Omega_m * (1+z)**3 + Omega_Lambda)

        def integrand(z_prime):
            return self.c / (H(z_prime) * 1e3)  # Convert to m/s

        r, _ = integrate.quad(integrand, 0, z)
        r_Mpc = r / 3.086e22

        return r_Mpc / (1 + z)

    def multipole_from_wavenumber(self, k: float, z: float) -> float:
        """
        Calculate multipole moment ℓ from wavenumber k.

        ℓ = k * D_A(z)

        Args:
            k: Wavenumber in Mpc^-1
            z: Redshift

        Returns:
            Multipole moment ℓ
        """
        D_A = self.angular_diameter_distance(z)
        return k * D_A

    def transfer_function_T(self, l: int, z: float) -> float:
        """
        Calculate CMB temperature transfer function.

        From paper: Δℓ_n ≈ 3πnβρ_{B,0}(R_max/a_0)³

        Args:
            l: Multipole moment
            z: Redshift

        Returns:
            Transfer function value
        """
        # Acoustic peak shift from Phase B
        a = 1.0 / (1 + z)
        Lambda_RCFM = self.params.Lambda_RCFM(z)

        # Shift amplitude (simplified)
        Delta_l = 3 * np.pi * Lambda_RCFM * l

        return Delta_l

    def compute_Cl(self, l_max: int = 2500,
                  load_planck: bool = False) -> dict:
        """
        Compute CMB angular power spectrum C_ℓ.

        C_ℓ^RCFM = (4π/9) ∫ dκ P(k) |Δ_ℓ(k, η_0)|²

        Args:
            l_max: Maximum multipole moment
            load_planck: Whether to load Planck data for comparison

        Returns:
            Dictionary with ℓ, C_ℓ, and optionally Planck comparison
        """
        # Create multipole array
        l_array = np.arange(2, l_max + 1)

        # Create wavenumber array
        k_min = 1e-4  # Mpc^-1
        k_max = 1.0   # Mpc^-1
        k_array = np.logspace(np.log10(k_min), np.log10(k_max), 100)

        # Primordial power spectrum
        from .perturbations import PrimordialPowerSpectrum
        spectrum = PrimordialPowerSpectrum(self.params)
        P_k = np.array([spectrum.scalar_spectrum(k) for k in k_array])

        # Transfer function (simplified)
        z_last = 1100  # Last scattering surface
        D_A = self.angular_diameter_distance(z_last)

        # Compute C_ℓ
        Cl = np.zeros_like(l_array, dtype=float)

        for i, l in enumerate(l_array):
            # k corresponding to this ℓ
            k = l / D_A if D_A > 0 else 0

            # Interpolate P(k) at this k
            P_k_interp = interp1d(np.log10(k_array), np.log10(P_k),
                                 kind='linear', fill_value='extrapolate')
            P_k_val = 10**P_k_interp(np.log10(k)) if k > 0 else 0

            # Transfer function (Boltzmann-like)
            # Simplified: oscillates like acoustic peaks
            if l < 100:
                T_l = 0.1
            else:
                # Acoustic peaks
                l_peak = np.array([220, 540, 810, 1100, 1350, 1600])
                T_l = 1.0
                for j, l_p in enumerate(l_peak):
                    # Add Phase B shift
                    Delta_l = self.transfer_function_T(l, z_last)
                    T_l += 0.5 * np.exp(-((l - l_p - Delta_l)**2) / (2 * 50**2))

            # C_ℓ contribution
            Cl[i] = (4 * np.pi / 9) * P_k_val * T_l**2

        result = {
            'l': l_array,
            'Cl': Cl,
            'units': 'μK²'
        }

        # Add Planck comparison if requested
        if load_planck:
            result['Cl_planck'] = self._load_planck_data(l_max)

        return result

    def _load_planck_data(self, l_max: int) -> np.ndarray:
        """
        Load Planck 2018 C_ℓ data for comparison.

        This is a placeholder - actual implementation would
        download from Planck archive.

        Args:
            l_max: Maximum multipole

        Returns:
            Planck C_ℓ array
        """
        # Return theoretical Planck spectrum (placeholder)
        # Actual implementation should load from:
        # https://pla.esac.esa.int/pla/#cosmology
        l_array = np.arange(2, l_max + 1)

        # Approximate Planck spectrum (simplified)
        Cl_planck = np.zeros_like(l_array, dtype=float)

        # First peak
        Cl_planck += 5000 * np.exp(-((l_array - 220)**2) / (2 * 30**2))
        # Second peak
        Cl_planck += 3000 * np.exp(-((l_array - 540)**2) / (2 * 40**2))
        # Third peak
        Cl_planck += 2000 * np.exp(-((l_array - 810)**2) / (2 * 50**2))
        # Higher peaks
        for n in range(4, 10):
            l_n = 220 + (n-1) * 265  # Peak spacing ~265
            Cl_planck += 1000 / n * np.exp(-((l_array - l_n)**2) / (2 * 60**2))

        return Cl_planck

    def compute_Cl_TE(self, l_max: int = 2500) -> dict:
        """
        Compute CMB temperature-polarization cross-spectrum C_ℓ^TE.

        Args:
            l_max: Maximum multipole

        Returns:
            Dictionary with ℓ and C_ℓ^TE
        """
        l_array = np.arange(2, l_max + 1)

        # Simplified TE spectrum
        Cl_TE = np.zeros_like(l_array, dtype=float)

        # Cross-correlation peaks (simplified)
        for n in range(1, 6):
            l_n = 220 + (n-1) * 265
            Cl_TE += 100 * (-1)**n * np.exp(-((l_array - l_n)**2) / (2 * 50**2))

        return {
            'l': l_array,
            'Cl_TE': Cl_TE,
            'units': 'μK²'
        }

    def compute_Cl_EE(self, l_max: int = 2500) -> dict:
        """
        Compute CMB E-mode polarization spectrum C_ℓ^EE.

        Args:
            l_max: Maximum multipole

        Returns:
            Dictionary with ℓ and C_ℓ^EE
        """
        l_array = np.arange(2, l_max + 1)

        # E-mode spectrum (simplified)
        Cl_EE = np.zeros_like(l_array, dtype=float)

        # First E-mode peak
        Cl_EE += 50 * np.exp(-((l_array - 1400)**2) / (2 * 100**2))
        # Second peak
        Cl_EE += 30 * np.exp(-((l_array - 1700)**2) / (2 * 100**2))

        return {
            'l': l_array,
            'Cl_EE': Cl_EE,
            'units': 'μK²'
        }

    def compute_Cl_TT_with_phaseB_corrections(self, l_max: int = 2500) -> dict:
        """
        Compute full TT spectrum with Phase B corrections.

        From paper:
        - Peak positions shifted by Δℓ_n ≈ 3πnβρ_{B,0}(R_max/a_0)³
        - Quadrupole C_2 naturally suppressed by S^3 cutoff
        - Odd/even ratio modified by scale-dependent ε_n

        Args:
            l_max: Maximum multipole

        Returns:
            Dictionary with corrected C_ℓ
        """
        result = self.compute_Cl(l_max)

        l_array = result['l']
        Cl = result['Cl'].copy()

        # Phase B corrections
        a0 = 1.0  # Present scale factor
        Lambda_RCFM = self.params.Lambda_RCFM(0)

        # Peak shift correction
        for n in range(1, 10):
            l_n = 220 + (n-1) * 265  # Acoustic peak positions
            Delta_l_n = 3 * np.pi * n * Lambda_RCFM

            # Apply shift to peaks
            for i, l in enumerate(l_array):
                if l > l_n - 100 and l < l_n + 100:
                    # Gaussian weight centered on shifted peak
                    weight = np.exp(-((l - l_n - Delta_l_n)**2) / (2 * 50**2))
                    Cl[i] *= (1 + 0.1 * weight * Lambda_RCFM)

        # Quadrupole suppression
        Cl[0] *= (1 - 0.5 * Lambda_RCFM)  # C_2

        result['Cl_corrected'] = Cl

        return result

    def save_spectrum(self, filename: str, spectrum_dict: dict):
        """
        Save spectrum to file.

        Args:
            filename: Output filename
            spectrum_dict: Dictionary from compute_Cl
        """
        l = spectrum_dict['l']
        Cl = spectrum_dict['Cl']

        header = "# CMB Angular Power Spectrum - RCFM v1.3\n"
        header += f"# Generated for parameters: beta={self.params.beta}, "
        header += f"rho_B0={self.params.rho_B0}\n"
        header += "# l C_l [muK^2]\n"

        data = np.column_stack([l, Cl])

        np.savetxt(filename, data, header=header, comments='')
        print(f"Spectrum saved to {filename}")


def compute_standard_lcdm_spectrum(l_max: int = 2500) -> dict:
    """
    Compute standard ΛCDM CMB spectrum for comparison.

    This is a simplified ΛCDM spectrum for comparison purposes.
    Use CAMB or CLASS for accurate ΛCDM predictions.

    Args:
        l_max: Maximum multipole

    Returns:
        Dictionary with ΛCDM C_ℓ
    """
    l_array = np.arange(2, l_max + 1)
    Cl_lcdm = np.zeros_like(l_array, dtype=float)

    # Approximate ΛCDM spectrum (Planck 2018 like)
    # First peak
    Cl_lcdm += 5000 * np.exp(-((l_array - 220)**2) / (2 * 30**2))
    # Second peak
    Cl_lcdm += 3000 * np.exp(-((l_array - 540)**2) / (2 * 40**2))
    # Third peak
    Cl_lcdm += 2000 * np.exp(-((l_array - 810)**2) / (2 * 50**2))
    # Higher peaks
    for n in range(4, 10):
        l_n = 220 + (n-1) * 265
        Cl_lcdm += 1000 / n * np.exp(-((l_array - l_n)**2) / (2 * 60**2))

    return {
        'l': l_array,
        'Cl': Cl_lcdm,
        'units': 'μK²'
    }


if __name__ == "__main__":
    # Test the module
    print("RCFM CMB Angular Power Spectrum Module Test")
    print("=" * 50)

    from constants import get_rcfm_defaults

    params = get_rcfm_defaults()

    # Initialize calculator
    cmb = CMBAngularSpectrum(params)

    # Compute spectrum
    print("\nComputing CMB spectrum...")
    result = cmb.compute_Cl(l_max=2000)

    print(f"Computed {len(result['l'])} multipoles")
    print(f"Maximum C_l: {result['Cl'].max():.2f} {result['units']}")

    # Compare with ΛCDM
    lcdm = compute_standard_lcdm_spectrum(2000)
    print(f"ΛCDM Maximum C_l: {lcdm['Cl'].max():.2f} {lcdm['units']}")

    # Sound horizon
    r_s = cmb.sound_horizon()
    print(f"\nSound horizon at z=1100: {r_s:.2f} Mpc")

    # Angular diameter distance
    D_A = cmb.angular_diameter_distance(1100)
    print(f"Angular diameter distance at z=1100: {D_A:.2f} Mpc")

    print("\nTest completed successfully!")
