"""
IPCC Tier 1 carbon calculation for mangrove ecosystems.

Converts mangrove area (hectares) to carbon credits (tCO2e) using default
constants from the IPCC 2013 Supplement to the 2006 Guidelines: Wetlands.

References:
    IPCC (2013). 2013 Supplement to the 2006 IPCC Guidelines for National
    Greenhouse Gas Inventories: Wetlands. Hiraishi, T., Krug, T., Tanabe, K.,
    Srivastava, N., Baasansuren, J., Fukuda, M. and Troxler, T.G. (eds).
    Published: IPCC, Switzerland.
"""

import numpy as np

# ---------------------------------------------------------------------------
# IPCC Wetlands Supplement (2013) default constants
# ---------------------------------------------------------------------------

# Aboveground biomass density for mangroves (Table 4.4)
# Source: IPCC 2013 Supplement to the 2006 Guidelines: Wetlands
BIOMASS_DENSITY = 230.0  # tonnes dry matter per hectare

# Carbon fraction of dry biomass (IPCC default)
# Source: IPCC 2013 Supplement to the 2006 Guidelines: Wetlands
CARBON_FRACTION = 0.47  # dimensionless

# Molecular weight ratio of CO2 to C: 44/12
# Source: IPCC 2013 Supplement to the 2006 Guidelines: Wetlands
CO2_TO_C_RATIO = 44.0 / 12.0  # ≈ 3.667

# Annual carbon sequestration rate for mangroves
# Source: IPCC 2013 Supplement to the 2006 Guidelines: Wetlands
ANNUAL_SEQUESTRATION = 7.0  # tCO2e per hectare per year


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def hectares_from_mask(mask: np.ndarray, pixel_size_m: float = 10.0) -> float:
    """Convert a binary segmentation mask to area in hectares.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask where 1 = mangrove, 0 = non-mangrove.
    pixel_size_m : float
        Ground sampling distance in metres (default 10 m for Sentinel-2).

    Returns
    -------
    float
        Mangrove area in hectares.
    """
    pixel_count = int(np.sum(mask == 1))
    pixel_area_m2 = pixel_size_m * pixel_size_m
    area_m2 = pixel_count * pixel_area_m2
    hectares = area_m2 / 10_000.0
    return hectares


def carbon_stock(hectares: float) -> dict:
    """Compute standing carbon stock from mangrove area.

    Uses the IPCC Tier 1 chain:
        biomass  = hectares × BIOMASS_DENSITY
        carbon   = biomass  × CARBON_FRACTION
        co2e     = carbon   × CO2_TO_C_RATIO

    Parameters
    ----------
    hectares : float
        Mangrove area in hectares.

    Returns
    -------
    dict
        Keys: hectares, biomass_t, carbon_t, co2e_t.
    """
    biomass = hectares * BIOMASS_DENSITY
    carbon = biomass * CARBON_FRACTION
    co2e = carbon * CO2_TO_C_RATIO

    return {
        "hectares": round(hectares, 4),
        "biomass_t": round(biomass, 4),
        "carbon_t": round(carbon, 4),
        "co2e_t": round(co2e, 4),
    }


def carbon_flux(
    baseline_hectares: float,
    current_hectares: float,
    years: int = 4,
) -> dict:
    """Compute carbon flux (change) between two time points.

    Parameters
    ----------
    baseline_hectares : float
        Mangrove area at baseline (e.g. 2020).
    current_hectares : float
        Mangrove area at current date (e.g. 2024).
    years : int
        Number of years between the two observations (default 4).

    Returns
    -------
    dict
        Keys: baseline_hectares, current_hectares, delta_hectares,
              annual_flux_tco2e, total_flux_tco2e, years.
    """
    delta_hectares = current_hectares - baseline_hectares
    annual_flux = delta_hectares * ANNUAL_SEQUESTRATION
    total_flux = annual_flux * years

    return {
        "baseline_hectares": round(baseline_hectares, 4),
        "current_hectares": round(current_hectares, 4),
        "delta_hectares": round(delta_hectares, 4),
        "annual_flux_tco2e": round(annual_flux, 4),
        "total_flux_tco2e": round(total_flux, 4),
        "years": years,
    }


def full_report(
    baseline_mask: np.ndarray,
    current_mask: np.ndarray,
    pixel_size_m: float = 10.0,
    years: int = 4,
) -> dict:
    """Generate a complete carbon report from two binary masks.

    Parameters
    ----------
    baseline_mask : np.ndarray
        Binary mask for the baseline year (e.g. 2020).
    current_mask : np.ndarray
        Binary mask for the current year (e.g. 2024).
    pixel_size_m : float
        Ground sampling distance in metres.
    years : int
        Number of years between baseline and current observations.

    Returns
    -------
    dict
        Combined report with baseline_stock, current_stock, and flux sections.
        Suitable for direct JSON serialization.
    """
    baseline_ha = hectares_from_mask(baseline_mask, pixel_size_m)
    current_ha = hectares_from_mask(current_mask, pixel_size_m)

    return {
        "baseline_stock": carbon_stock(baseline_ha),
        "current_stock": carbon_stock(current_ha),
        "flux": carbon_flux(baseline_ha, current_ha, years),
    }


def print_report(report: dict) -> None:
    """Pretty-print a full carbon report to the console.

    Parameters
    ----------
    report : dict
        Output of :func:`full_report`.
    """
    def _print_stock(label: str, stock: dict) -> None:
        print(f"  {label}:")
        print(f"    Area              : {stock['hectares']:>12.2f} ha")
        print(f"    Biomass           : {stock['biomass_t']:>12.2f} t")
        print(f"    Carbon            : {stock['carbon_t']:>12.2f} t C")
        print(f"    CO2 equivalent    : {stock['co2e_t']:>12.2f} tCO2e")

    print("=" * 60)
    print("  IPCC Tier 1 Mangrove Carbon Report")
    print("  Source: IPCC 2013 Supplement to the 2006 Guidelines: Wetlands")
    print("=" * 60)

    _print_stock("Baseline Stock", report["baseline_stock"])
    print()
    _print_stock("Current Stock", report["current_stock"])

    flux = report["flux"]
    print()
    print("  Carbon Flux:")
    print(f"    Delta area        : {flux['delta_hectares']:>12.2f} ha")
    print(f"    Annual flux       : {flux['annual_flux_tco2e']:>12.2f} tCO2e/yr")
    print(f"    Total flux ({flux['years']}yr)  : {flux['total_flux_tco2e']:>12.2f} tCO2e")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic example: 1000×1000 masks, 10 m pixels
    rng = np.random.default_rng(42)
    baseline = (rng.random((1000, 1000)) < 0.02).astype(np.uint8)
    current = (rng.random((1000, 1000)) < 0.025).astype(np.uint8)

    report = full_report(baseline, current, pixel_size_m=10.0, years=4)
    print_report(report)
