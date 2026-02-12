"""
Range-constrained pedotransfer functions for peat soils.

Defines allowable (min, max) ranges for MVG parameters at discrete
bulk-density breakpoints and provides helpers to:

* interpolate those ranges at any BD value,
* clamp (or validate) PTF-predicted parameters so they stay within bounds.

The ranges are expressed in **log10 space** for Ks, α, and n, matching the
regression form used by ``ptf_all_types_high_bd``.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from peat_ptf import MVGParameters, ptf_all_types_high_bd

# ---------------------------------------------------------------------------
# Allowable ranges at discrete BD breakpoints  (Table-derived bounds)
# ---------------------------------------------------------------------------
# Each list has one entry per BD breakpoint.  The tuple is (min, max).

bulk_density_range: list[float] = [0.2, 0.4, 0.6, 0.8]

log10_ks_range: list[tuple[float, float]] = [
    (-2.4, 1.0),
    (-2.0, 0.9),
    (-1.0, 0.5),
    (-0.8, 0.0),
]

log10_n_range: list[tuple[float, float]] = [
    (0.02, 0.20),
    (0.02, 0.15),
    (0.03, 0.12),
    (0.03, 0.10),
]

log10_alpha_range: list[tuple[float, float]] = [
    (-2.7, -0.9),
    (-2.5, -1.0),
    (-2.1, -0.9),
    (-2.0, -1.0),
]

tau_range: list[tuple[float, float]] = [
    (0.5, 0.5),
    (0.5, 0.5),
    (0.5, 0.5),
    (0.5, 0.5),
]


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def _interp_range(
    BD: float,
    breakpoints: list[float],
    ranges: list[tuple[float, float]],
) -> tuple[float, float]:
    """Linearly interpolate (min, max) bounds at *BD* from discrete breakpoints.

    * If *BD* ≤ smallest breakpoint → returns the first range entry.
    * If *BD* ≥ largest breakpoint  → returns the last range entry.
    * Otherwise, linearly interpolates both the *min* and *max* values
      between the two neighbouring breakpoints.
    """
    if BD <= breakpoints[0]:
        return ranges[0]
    if BD >= breakpoints[-1]:
        return ranges[-1]

    # Locate surrounding breakpoints
    for i in range(len(breakpoints) - 1):
        if breakpoints[i] <= BD <= breakpoints[i + 1]:
            frac = (BD - breakpoints[i]) / (breakpoints[i + 1] - breakpoints[i])
            lo = ranges[i][0] + frac * (ranges[i + 1][0] - ranges[i][0])
            hi = ranges[i][1] + frac * (ranges[i + 1][1] - ranges[i][1])
            return (lo, hi)

    # Fallback (should not be reached)
    return ranges[-1]


def interpolate_ranges(BD: float) -> Dict[str, Tuple[float, float]]:
    """Return interpolated (min, max) bounds for each parameter at *BD*.

    Keys in the returned dict:

    * ``"log10_Ks"`` – bounds in log10(cm h⁻¹)
    * ``"log10_alpha"`` – bounds in log10(cm⁻¹)
    * ``"log10_n"`` – bounds in log10(dimensionless)
    * ``"tau"`` – bounds in linear space

    Parameters
    ----------
    BD : float
        Bulk density (g cm⁻³).  Typically > 0.2 (the domain of
        ``ptf_all_types_high_bd``).

    Returns
    -------
    dict[str, tuple[float, float]]
    """
    return {
        "log10_Ks":    _interp_range(BD, bulk_density_range, log10_ks_range),
        "log10_alpha": _interp_range(BD, bulk_density_range, log10_alpha_range),
        "log10_n":     _interp_range(BD, bulk_density_range, log10_n_range),
        "tau":         _interp_range(BD, bulk_density_range, tau_range),
    }


# ---------------------------------------------------------------------------
# Clamping / validation
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the interval [lo, hi]."""
    return max(lo, min(hi, value))


def constrain_parameters(
    BD: float,
    params: Optional[MVGParameters] = None,
    *,
    mode: str = "clamp",
) -> MVGParameters:
    """Predict MVG parameters with ``ptf_all_types_high_bd`` and constrain
    them to the allowable ranges interpolated at *BD*.

    Parameters
    ----------
    BD : float
        Bulk density (g cm⁻³).  Should be > 0.2 (the valid domain of
        ``ptf_all_types_high_bd``).
    params : MVGParameters or None, optional
        Pre-computed parameters.  If *None*, ``ptf_all_types_high_bd(BD)``
        is called automatically.
    mode : ``"clamp"`` | ``"validate"``
        * ``"clamp"`` (default) – silently clamp each parameter to its
          interpolated [min, max] range.
        * ``"validate"`` – raise ``ValueError`` listing every parameter
          that falls outside its allowable range.

    Returns
    -------
    MVGParameters
        The (possibly clamped) parameter set.  ``theta_s`` and ``theta_r``
        are left unchanged (no range defined for them).

    Raises
    ------
    ValueError
        If *mode* is ``"validate"`` and any parameter is out of range.
    """
    if mode not in ("clamp", "validate"):
        raise ValueError(f"mode must be 'clamp' or 'validate', got '{mode}'")

    if params is None:
        params = ptf_all_types_high_bd(BD)

    bounds = interpolate_ranges(BD)

    # Current values in log10 space
    log10_Ks    = math.log10(params.Ks)
    log10_alpha = math.log10(params.alpha)
    log10_n     = math.log10(params.n)
    tau         = params.tau

    if mode == "validate":
        violations: list[str] = []
        for name, val, (lo, hi) in [
            ("log10_Ks",    log10_Ks,    bounds["log10_Ks"]),
            ("log10_alpha", log10_alpha, bounds["log10_alpha"]),
            ("log10_n",     log10_n,     bounds["log10_n"]),
            ("tau",         tau,         bounds["tau"]),
        ]:
            if val < lo or val > hi:
                violations.append(
                    f"{name}={val:.4f} outside [{lo:.4f}, {hi:.4f}]"
                )
        if violations:
            raise ValueError(
                "Parameter(s) outside allowable range at "
                f"BD={BD}: " + "; ".join(violations)
            )
        return params  # all within range – return unchanged

    # mode == "clamp"
    log10_Ks    = _clamp(log10_Ks,    *bounds["log10_Ks"])
    log10_alpha = _clamp(log10_alpha, *bounds["log10_alpha"])
    log10_n     = _clamp(log10_n,     *bounds["log10_n"])
    tau         = _clamp(tau,         *bounds["tau"])

    return MVGParameters(
        theta_s=params.theta_s,
        alpha=10 ** log10_alpha,
        n=10 ** log10_n,
        Ks=10 ** log10_Ks,
        tau=tau,
        theta_r=params.theta_r,
    )


# ---------------------------------------------------------------------------
# Exploring the feasible parameter space
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: a at t=0, b at t=1."""
    return a + t * (b - a)


def parameter_at_fraction(
    BD: float,
    fraction: float = 0.5,
    *,
    theta_s: Optional[float] = None,
    theta_r: float = 0.0,
) -> MVGParameters:
    """Return an MVG parameter set at a given *fraction* within the allowable
    range for each parameter.

    ``fraction = 0.0`` gives the lower bounds (smallest Ks, α, n),
    ``fraction = 1.0`` gives the upper bounds, and ``fraction = 0.5``
    gives the mid-range values.  This lets you quickly generate
    "easy-to-simulate" (smoother, lower-Ks) or "hard" (steep, high-Ks)
    parameter sets while staying within physically plausible limits.

    Parameters
    ----------
    BD : float
        Bulk density (g cm⁻³).
    fraction : float
        Position within each range, 0 → lower bound, 1 → upper bound.
    theta_s : float or None
        Saturated water content.  If *None*, computed as
        ``0.950 − 0.437 × BD`` (the standard regression).
    theta_r : float
        Residual water content (default 0.0).

    Returns
    -------
    MVGParameters
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")

    bounds = interpolate_ranges(BD)
    if theta_s is None:
        theta_s = 0.950 - 0.437 * BD

    log10_Ks    = _lerp(*bounds["log10_Ks"],    fraction)
    log10_alpha = _lerp(*bounds["log10_alpha"],  fraction)
    log10_n     = _lerp(*bounds["log10_n"],      fraction)
    tau         = _lerp(*bounds["tau"],           fraction)

    return MVGParameters(
        theta_s=theta_s,
        alpha=10 ** log10_alpha,
        n=10 ** log10_n,
        Ks=10 ** log10_Ks,
        tau=tau,
        theta_r=theta_r,
    )


def parameter_at_fraction_inv(
    BD: float,
    fraction: float = 0.5,
    *,
    theta_s: Optional[float] = None,
    theta_r: float = 0.0,
) -> MVGParameters:
    """Like :func:`parameter_at_fraction`, but **α and n move in opposite
    directions** relative to the fraction.

    When *fraction* increases from 0 → 1:

    * **Ks** and **τ** increase (low → high), same as the normal version.
    * **α decreases** (high → low) — i.e. uses ``1 − fraction``.
    * **n increases** (low → high) — same direction as normal.

    This produces the physically common pairing of a **steep retention curve**
    (high n) with a **low air-entry value** (low α), and vice versa.  It
    gives you a complementary axis of variability to explore when searching
    for parameter sets that your simulation software can handle.

    Parameters
    ----------
    BD : float
        Bulk density (g cm⁻³).
    fraction : float
        Position within each range, 0 → 1.  α is inverted (uses 1 − fraction).
    theta_s : float or None
        Saturated water content.  If *None*, computed from the standard
        regression ``0.950 − 0.437 × BD``.
    theta_r : float
        Residual water content (default 0.0).

    Returns
    -------
    MVGParameters
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")

    bounds = interpolate_ranges(BD)
    if theta_s is None:
        theta_s = 0.950 - 0.437 * BD

    log10_Ks    = _lerp(*bounds["log10_Ks"],    fraction)
    log10_alpha = _lerp(*bounds["log10_alpha"],  1.0 - fraction)  # inverted
    log10_n     = _lerp(*bounds["log10_n"],      fraction)
    tau         = _lerp(*bounds["tau"],           fraction)

    return MVGParameters(
        theta_s=theta_s,
        alpha=10 ** log10_alpha,
        n=10 ** log10_n,
        Ks=10 ** log10_Ks,
        tau=tau,
        theta_r=theta_r,
    )


def sample_parameter_sets(
    BD: float,
    n_sets: int = 5,
    *,
    theta_s: Optional[float] = None,
    theta_r: float = 0.0,
) -> list[MVGParameters]:
    """Generate *n_sets* MVG parameter sets evenly spaced across the
    allowable range at *BD*.

    Useful for giving your simulation software several options to choose
    from — spanning the full physically-plausible envelope for that
    bulk density.

    Parameters
    ----------
    BD : float
        Bulk density (g cm⁻³).
    n_sets : int
        Number of parameter sets to generate (≥ 1).
    theta_s, theta_r
        See :func:`parameter_at_fraction`.

    Returns
    -------
    list[MVGParameters]
        Ordered from lower-bound to upper-bound values.
    """
    if n_sets < 1:
        raise ValueError(f"n_sets must be ≥ 1, got {n_sets}")
    if n_sets == 1:
        return [parameter_at_fraction(BD, 0.5, theta_s=theta_s, theta_r=theta_r)]

    fractions = [i / (n_sets - 1) for i in range(n_sets)]
    return [
        parameter_at_fraction(BD, f, theta_s=theta_s, theta_r=theta_r)
        for f in fractions
    ]


def named_parameter_sets(
    BD: float,
    *,
    theta_s: Optional[float] = None,
    theta_r: float = 0.0,
) -> Dict[str, MVGParameters]:
    """Return three labelled parameter sets: ``"low"``, ``"mid"``, ``"high"``.

    * **low**  (fraction = 0.0) – lower bounds; typically the easiest for
      numerical solvers (low Ks, gentle retention curve).
    * **mid**  (fraction = 0.5) – centre of the range.
    * **high** (fraction = 1.0) – upper bounds; steeper curves, higher Ks.

    Parameters
    ----------
    BD : float
        Bulk density (g cm⁻³).
    theta_s, theta_r
        See :func:`parameter_at_fraction`.

    Returns
    -------
    dict[str, MVGParameters]
    """
    return {
        "low":  parameter_at_fraction(BD, 0.0, theta_s=theta_s, theta_r=theta_r),
        "mid":  parameter_at_fraction(BD, 0.5, theta_s=theta_s, theta_r=theta_r),
        "high": parameter_at_fraction(BD, 1.0, theta_s=theta_s, theta_r=theta_r),
    }


def get_range_summary(BD: float) -> Dict[str, Dict[str, float]]:
    """Return the allowable ranges at *BD* in **linear** (physical) units.

    Handy for printing or feeding into a table.

    Returns
    -------
    dict
        Nested dict with keys ``"Ks"``, ``"alpha"``, ``"n"``, ``"tau"``,
        each containing ``{"min": …, "max": …}``.
    """
    bounds = interpolate_ranges(BD)
    return {
        "Ks": {
            "min": 10 ** bounds["log10_Ks"][0],
            "max": 10 ** bounds["log10_Ks"][1],
        },
        "alpha": {
            "min": 10 ** bounds["log10_alpha"][0],
            "max": 10 ** bounds["log10_alpha"][1],
        },
        "n": {
            "min": 10 ** bounds["log10_n"][0],
            "max": 10 ** bounds["log10_n"][1],
        },
        "tau": {
            "min": bounds["tau"][0],
            "max": bounds["tau"][1],
        },
    }

