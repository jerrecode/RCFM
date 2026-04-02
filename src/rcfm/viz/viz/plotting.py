#!/usr/bin/env python3
"""
RCFM Visualization Module - Plotting Scripts

This module contains visualization functions for RCFM results.

Author: MiniMax Agent
Version: 1.3
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import os


def setup_plotting():
    """Setup plotting style for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'text.usetex': False,  # Set to True for LaTeX rendering
    })


def plot_background_evolution(r: np.ndarray,
                            a: np.ndarray,
                            H: np.ndarray,
                            rho_A: np.ndarray,
                            rho_B: np.ndarray,
                            save_path: Optional[str] = None):
    """
    Plot background evolution.

    Args:
        r: Radial coordinates
        a: Scale factor
        H: Hubble parameter
        rho_A: Phase A density
        rho_B: Phase B density
        save_path: Path to save figure
    """
    setup_plotting()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Scale factor
    axes[0, 0].plot(r / r.max(), a, 'b-', linewidth=2)
    axes[0, 0].set_xlabel(r'$r/R_{\max}$')
    axes[0, 0].set_ylabel(r'$a(r)$')
    axes[0, 0].set_title('Scale Factor Evolution')
    axes[0, 0].grid(True, alpha=0.3)

    # Hubble parameter
    axes[0, 1].plot(r / r.max(), H * 3.086e22, 'r-', linewidth=2)  # Convert to km/s/Mpc
    axes[0, 1].set_xlabel(r'$r/R_{\max}$')
    axes[0, 1].set_ylabel(r'$H(r)$ [km/s/Mpc]')
    axes[0, 1].set_title('Hubble Parameter Evolution')
    axes[0, 1].grid(True, alpha=0.3)

    # Phase A density
    axes[1, 0].plot(r / r.max(), rho_A, 'g-', linewidth=2)
    axes[1, 0].set_xlabel(r'$r/R_{\max}$')
    axes[1, 0].set_ylabel(r'$\rho_A(r)$ [kg/m³]')
    axes[1, 0].set_title('Phase A Density')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # Phase B density
    axes[1, 1].plot(r / r.max(), rho_B, 'm-', linewidth=2)
    axes[1, 1].set_xlabel(r'$r/R_{\max}$')
    axes[1, 1].set_ylabel(r'$\rho_B(r)$ [kg/m³]')
    axes[1, 1].set_title('Phase B Density')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_H_of_z(H_rcfm: np.ndarray,
               z: np.ndarray,
               H_planck: np.ndarray,
               save_path: Optional[str] = None):
    """
    Plot H(z) comparison with Planck data.

    Args:
        H_rcfm: RCFM H(z)
        z: Redshift array
        H_planck: Planck H(z) data
        save_path: Path to save figure
    """
    setup_plotting()

    fig, ax = plt.subplots(figsize=(10, 6))

    # RCFM prediction
    ax.plot(z, H_rcfm * 3.086e22, 'b-', linewidth=2, label='RCFM')

    # Planck data
    ax.errorbar(H_planck[:, 0], H_planck[:, 1],
                yerr=H_planck[:, 2],
                fmt='o', color='red', markersize=6,
                capsize=3, label='Planck 2018')

    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$H(z)$ [km/s/Mpc]')
    ax.set_title('Hubble Parameter Evolution: RCFM vs Planck')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_matter_power_spectrum(k: np.ndarray,
                               Pk_rcfm: np.ndarray,
                               Pk_lcdm: np.ndarray,
                               save_path: Optional[str] = None):
    """
    Plot matter power spectrum.

    Args:
        k: Wavenumber
        Pk_rcfm: RCFM power spectrum
        Pk_lcdm: LCDM power spectrum
        save_path: Path to save figure
    """
    setup_plotting()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(k, Pk_rcfm, 'b-', linewidth=2, label='RCFM')
    ax.plot(k, Pk_lcdm, 'r--', linewidth=2, label=r'$\Lambda$CDM')

    ax.set_xlabel(r'$k$ [h/Mpc]')
    ax.set_ylabel(r'$P(k)$ [Mpc³/h³]')
    ax.set_title('Matter Power Spectrum')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_cmb_spectrum(l: np.ndarray,
                     Cl_rcfm: np.ndarray,
                     Cl_planck: np.ndarray,
                     save_path: Optional[str] = None):
    """
    Plot CMB angular power spectrum.

    Args:
        l: Multipole moments
        Cl_rcfm: RCFM C_l
        Cl_planck: Planck C_l
        save_path: Path to save figure
    """
    setup_plotting()

    fig, ax = plt.subplots(figsize=(10, 6))

    l_factor = l * (l + 1) / (2 * np.pi)

    ax.plot(l, Cl_rcfm * l_factor, 'b-', linewidth=2, label='RCFM')
    ax.plot(l, Cl_planck * l_factor, 'r--', linewidth=2, label='Planck 2018')

    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\ell(\ell+1)C_\ell/2\pi$ [$\mu$K²]')
    ax.set_title('CMB Temperature Power Spectrum')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_gw_stochastic_background(f: np.ndarray,
                                 Omega_gw: np.ndarray,
                                 save_path: Optional[str] = None):
    """
    Plot stochastic gravitational wave background.

    Args:
        f: Frequency
        Omega_gw: GW energy density spectrum
        save_path: Path to save figure
    """
    setup_plotting()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(f, Omega_gw, 'b-', linewidth=2)

    ax.set_xlabel(r'$f$ [Hz]')
    ax.set_ylabel(r'$\Omega_{\rm GW}(f)$')
    ax.set_title('Stochastic Gravitational Wave Background')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Add sensitivity curves (simplified)
    # LISA
    ax.fill_between([1e-3, 1e-1], 1e-12, 1e-8, alpha=0.2, color='green', label='LISA')
    # PTA
    ax.fill_between([1e-9, 1e-7], 1e-10, 1e-6, alpha=0.2, color='orange', label='PTA')

    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_parameter_constraints(chain_file: str,
                              save_path: Optional[str] = None):
    """
    Plot MCMC parameter constraints.

    Args:
        chain_file: Path to MCMC chain file
        save_path: Path to save figure
    """
    # This is a placeholder - actual implementation would use getdist or emcee
    print("Parameter constraint plotting not yet implemented")
    pass


if __name__ == "__main__":
    # Test plotting functions
    print("RCFM Visualization Module Test")
    print("=" * 50)

    # Generate test data
    r = np.linspace(0.01, 1.0, 100)
    a = r**2
    H = np.ones_like(r) * 70 * 3.086e22 / 1000  # km/s/Mpc
    rho_A = 1e-26 * r**(-3)
    rho_B = 1e-26 * np.exp(-r)

    # Test background evolution plot
    plot_background_evolution(r, a, H, rho_A, rho_B)
    print("Background evolution plot created successfully")
