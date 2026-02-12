"""
Pedotransfer functions (PTFs) for estimating Mualem-van Genuchten (MVG)
hydraulic parameters of peat soils.

Based on:
  - Table 4: Linear regression PTFs where hydraulic parameters (Ks, α, n, τ)
    are predicted from bulk density (BD), organic matter content (OM), and depth.
  - Table 5: Representative MVG parameter sets for subgroups of peat soils.

Units:
  - θs: saturated water content (cm³ cm⁻³)
  - Ks: saturated hydraulic conductivity (cm h⁻¹)
  - α:  van Genuchten alpha (cm⁻¹)
  - n:  van Genuchten n (dimensionless)
  - τ:  Mualem tortuosity parameter (dimensionless)
  - BD: bulk density (g cm⁻³)
  - OM: organic matter content (wt%)
  - depth: sample depth (cm)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike
else:
    try:
        import numpy as np
        from numpy.typing import ArrayLike
    except ImportError:  # numpy is optional for core PTF functions
        np = None  # type: ignore[assignment]
        ArrayLike = Any  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Data class for MVG parameters
# ---------------------------------------------------------------------------

@dataclass
class MVGParameters:
    """Mualem-van Genuchten hydraulic parameter set.

    The residual water content ``theta_r`` defaults to **0.0** for peat soils,
    which is the standard assumption in the literature (e.g. Schwärzel et al.,
    2006; Liu & Lennartz, 2019).  Override it when a measured value is
    available.
    """
    theta_s: float   # saturated water content (cm³ cm⁻³)
    alpha: float     # van Genuchten α (cm⁻¹)
    n: float         # van Genuchten n (dimensionless)
    Ks: float        # saturated hydraulic conductivity (cm h⁻¹)
    tau: float       # Mualem tortuosity τ (dimensionless)
    theta_r: float = 0.0  # residual water content (cm³ cm⁻³)

    @property
    def m(self) -> float:
        """van Genuchten *m* parameter (``1 − 1/n``)."""
        return 1.0 - 1.0 / self.n

    def __repr__(self) -> str:
        return (
            f"MVGParameters(θs={self.theta_s:.4f}, θr={self.theta_r:.4f}, "
            f"α={self.alpha:.4f}, n={self.n:.4f}, "
            f"Ks={self.Ks:.4f}, τ={self.tau:.4f})"
        )


# ---------------------------------------------------------------------------
# Table 4 – Pedotransfer functions (regression equations)
# ---------------------------------------------------------------------------

def ptf_sphagnum(BD: float, OM: float, depth: float) -> MVGParameters:
    """
    PTF for **Sphagnum** peat with BD ≤ 0.2 g cm⁻³.

    Parameters
    ----------
    BD : float
        Bulk density (g cm⁻³).  Must be ≤ 0.2.
    OM : float
        Organic matter content (wt%).
    depth : float
        Sample depth (cm).

    Returns
    -------
    MVGParameters
    """
    if BD > 0.2:
        raise ValueError(f"Sphagnum PTF valid for BD ≤ 0.2; got BD={BD}")

    # θs – not provided for Sphagnum individually; use the "All data" equation
    theta_s = 0.950 - 0.437 * BD

    # log10(Ks) = 3.362 − 55.113 × BD + 172.728 × BD²
    log_Ks = 3.362 - 55.113 * BD + 172.728 * BD ** 2
    Ks = 10 ** log_Ks

    # log10(α) = 4.497 − 7.493 × BD − 0.046 × OM − 0.021 × Depth
    log_alpha = 4.497 - 7.493 * BD - 0.046 * OM - 0.021 * depth
    alpha = 10 ** log_alpha

    # log10(n) = 0.182 − 0.714 × BD
    log_n = 0.182 - 0.714 * BD
    n = 10 ** log_n

    # τ = −5.086 + 67.880 × BD
    tau = -5.086 + 67.880 * BD

    return MVGParameters(theta_s=theta_s, alpha=alpha, n=n, Ks=Ks, tau=tau)


def ptf_woody(BD: float, depth: float, OM: float) -> MVGParameters:
    """
    PTF for **Woody** peat with BD ≤ 0.2 g cm⁻³.

    Parameters
    ----------
    BD : float
        Bulk density (g cm⁻³).  Must be ≤ 0.2.
    depth : float
        Sample depth (cm).
    OM : float
        Organic matter content (wt%).

    Returns
    -------
    MVGParameters
    """
    if BD > 0.2:
        raise ValueError(f"Woody PTF valid for BD ≤ 0.2; got BD={BD}")

    # θs – use "All data" equation
    theta_s = 0.950 - 0.437 * BD

    # log10(Ks) = 3.538 − 26.542 × BD
    log_Ks = 3.538 - 26.542 * BD
    Ks = 10 ** log_Ks

    # log10(α) = 2.799 − 18.846 × BD − 0.027 × Depth
    log_alpha = 2.799 - 18.846 * BD - 0.027 * depth
    alpha = 10 ** log_alpha

    # log10(n) = 0.634 − 0.006 × OM
    log_n = 0.634 - 0.006 * OM
    n = 10 ** log_n

    # τ = −4.84  (constant)
    tau = -4.84

    return MVGParameters(theta_s=theta_s, alpha=alpha, n=n, Ks=Ks, tau=tau)


def ptf_sedge(BD: float, depth: float) -> MVGParameters:
    """
    PTF for **Sedge** peat with BD ≤ 0.2 g cm⁻³.

    Parameters
    ----------
    BD : float
        Bulk density (g cm⁻³).  Must be ≤ 0.2.
    depth : float
        Sample depth (cm).

    Returns
    -------
    MVGParameters
    """
    if BD > 0.2:
        raise ValueError(f"Sedge PTF valid for BD ≤ 0.2; got BD={BD}")

    # θs – use "All data" equation
    theta_s = 0.950 - 0.437 * BD

    # log10(Ks) = 1.561 − 9.432 × BD
    log_Ks = 1.561 - 9.432 * BD
    Ks = 10 ** log_Ks

    # log10(α) = −0.575 − 13.441 × BD² − 0.011 × Depth
    log_alpha = -0.575 - 13.441 * BD ** 2 - 0.011 * depth
    alpha = 10 ** log_alpha

    # log10(n) = 0.124 − 1.766 × BD²
    log_n = 0.124 - 1.766 * BD ** 2
    n = 10 ** log_n

    # τ = 0.5  (constant)
    tau = 0.5

    return MVGParameters(theta_s=theta_s, alpha=alpha, n=n, Ks=Ks, tau=tau)


def ptf_all_types_high_bd(BD: float) -> MVGParameters:
    """
    PTF for **all peat types** with BD > 0.2 g cm⁻³ (up to 0.76 g cm⁻³).

    Parameters
    ----------
    BD : float
        Bulk density (g cm⁻³).  Must be > 0.2 (and ≤ 0.76).

    Returns
    -------
    MVGParameters
    """
    if BD <= 0.2:
        raise ValueError(
            f"All-types (high BD) PTF valid for BD > 0.2; got BD={BD}"
        )

    # θs
    theta_s = 0.950 - 0.437 * BD

    # log10(Ks) = 1.935 − 15.802 × BD + 19.552 × BD²
    log_Ks = 1.935 - 15.802 * BD + 19.552 * BD ** 2
    Ks = 10 ** log_Ks

    # log10(α) = −1.994 + 1.191 × BD²
    log_alpha = -1.994 + 1.191 * BD ** 2
    alpha = 10 ** log_alpha

    # log10(n) = 0.089 − 0.088 × BD²
    log_n = 0.089 - 0.088 * BD ** 2
    n = 10 ** log_n

    # τ = 0.5  (constant)
    tau = 0.5

    return MVGParameters(theta_s=theta_s, alpha=alpha, n=n, Ks=Ks, tau=tau)


def ptf_all_data(
    BD: float,
    depth: float = 0.0,
    OM: Optional[float] = None,
    peat_type: Optional[str] = None,
) -> MVGParameters:
    """
    General PTF for **all peat data** (BD ≤ 0.76 g cm⁻³).

    This dispatches to the peat-type-specific PTF when *peat_type* is
    provided and BD ≤ 0.2.  Otherwise it uses the "All data" equations
    from Table 4.

    Parameters
    ----------
    BD : float
        Bulk density (g cm⁻³).
    depth : float, optional
        Sample depth (cm).  Default 0.
    OM : float or None, optional
        Organic matter content (wt%).  Required for Sphagnum and Woody PTFs.
    peat_type : str or None, optional
        One of ``"sphagnum"``, ``"woody"``, ``"sedge"``.
        If ``None``, the general "All data" equations are used.

    Returns
    -------
    MVGParameters
    """
    # Dispatch to type-specific PTFs when applicable
    if peat_type is not None and BD <= 0.2:
        ptype = peat_type.strip().lower()
        if ptype == "sphagnum":
            if OM is None:
                raise ValueError("OM is required for Sphagnum PTF")
            return ptf_sphagnum(BD, OM, depth)
        elif ptype == "woody":
            if OM is None:
                raise ValueError("OM is required for Woody PTF")
            return ptf_woody(BD, depth, OM)
        elif ptype == "sedge":
            return ptf_sedge(BD, depth)
        else:
            raise ValueError(
                f"Unknown peat_type '{peat_type}'; "
                "expected 'sphagnum', 'woody', or 'sedge'."
            )

    if BD > 0.2:
        return ptf_all_types_high_bd(BD)

    # "All data" equations (BD ≤ 0.76, no peat type specified, BD ≤ 0.2)
    # θs = 0.950 − 0.437 × BD
    theta_s = 0.950 - 0.437 * BD

    # log10(Ks) = 1.935 − 15.802 × BD + 19.552 × BD²
    log_Ks = 1.935 - 15.802 * BD + 19.552 * BD ** 2
    Ks = 10 ** log_Ks

    # log10(α) = 0.326 − 9.135 × BD + 10.420 × BD² − 0.014 × Depth
    log_alpha = 0.326 - 9.135 * BD + 10.420 * BD ** 2 - 0.014 * depth
    alpha = 10 ** log_alpha

    # log10(n) = 0.153 − 0.422 × BD + 0.450 × BD²
    log_n = 0.153 - 0.422 * BD + 0.450 * BD ** 2
    n = 10 ** log_n

    # τ = −3.024 + 7.242 × BD
    tau = -3.024 + 7.242 * BD

    return MVGParameters(theta_s=theta_s, alpha=alpha, n=n, Ks=Ks, tau=tau)


# ---------------------------------------------------------------------------
# Table 5 – Representative grouped MVG parameter sets
# ---------------------------------------------------------------------------

def mvg_grouped_sphagnum(BD: float) -> MVGParameters:
    """
    Return representative MVG parameters for **Sphagnum** peat based on
    bulk-density group (Table 5).

    Groups
    ------
    I   : BD ≤ 0.05
    II  : 0.05 < BD ≤ 0.05  (transition)
    III : 0.05 < BD ≤ 0.1
    IV  : 0.1  < BD ≤ 0.2
    V   : BD > 0.2
    """
    if BD <= 0.05:
        # Group I – MVG
        return MVGParameters(theta_s=0.93, alpha=1.393, n=1.32, Ks=576.26, tau=-4.40)
    elif BD <= 0.1:
        # Group III – MVG
        return MVGParameters(theta_s=0.93, alpha=0.163, n=1.25, Ks=718.92, tau=-1.79)
    elif BD <= 0.2:
        # Group IV – MVG
        return MVGParameters(theta_s=0.91, alpha=0.222, n=1.21, Ks=math.nan, tau=math.nan)
    else:
        # Group V – MVG
        return MVGParameters(theta_s=0.91, alpha=0.020, n=1.28, Ks=math.nan, tau=math.nan)


def mvg_grouped_sphagnum_ptf(BD: float) -> MVGParameters:
    """PTF-derived grouped parameters for Sphagnum peat (Table 5)."""
    if BD <= 0.05:
        # Group I – PTF
        return MVGParameters(theta_s=0.94, alpha=0.784, n=1.45, Ks=58.41, tau=-2.90)
    elif BD <= 0.1:
        # Group III – PTF
        return MVGParameters(theta_s=0.92, alpha=0.184, n=1.34, Ks=4.13, tau=-0.89)
    elif BD <= 0.2:
        # Group IV – PTF
        return MVGParameters(theta_s=0.89, alpha=0.364, n=1.23, Ks=math.nan, tau=math.nan)
    else:
        # Group V – PTF
        return MVGParameters(theta_s=0.89, alpha=0.040, n=1.23, Ks=math.nan, tau=math.nan)


def mvg_grouped_woody(BD: float) -> MVGParameters:
    """
    Return representative **MVG-optimised** parameters for **Woody** peat
    based on bulk-density group (Table 5).

    Groups
    ------
    I   : BD ≤ 0.1
    II  : 0.1 < BD (transition)
    III : 0.1 < BD ≤ 0.2
    IV  : BD > 0.2
    """
    if BD <= 0.1:
        # Group I – MVG
        return MVGParameters(theta_s=0.96, alpha=21.38, n=1.26, Ks=40.12, tau=-4.69)
    elif BD <= 0.2:
        # Group III – MVG
        return MVGParameters(theta_s=0.92, alpha=2.518, n=1.26, Ks=20.74, tau=-5.66)
    else:
        # Group IV – MVG
        return MVGParameters(theta_s=0.87, alpha=0.062, n=1.20, Ks=math.nan, tau=math.nan)


def mvg_grouped_woody_ptf(BD: float) -> MVGParameters:
    """PTF-derived grouped parameters for Woody peat (Table 5)."""
    if BD <= 0.1:
        # Group I – PTF
        return MVGParameters(theta_s=0.92, alpha=23.52, n=1.23, Ks=55.77, tau=-4.84)
    elif BD <= 0.2:
        # Group III – PTF
        return MVGParameters(theta_s=0.90, alpha=0.729, n=1.25, Ks=0.49, tau=-4.84)
    else:
        # Group IV – PTF
        return MVGParameters(theta_s=0.88, alpha=0.069, n=1.25, Ks=math.nan, tau=math.nan)


def mvg_grouped_sedge(BD: float) -> MVGParameters:
    """
    Return representative **MVG-optimised** parameters for **Sedge** peat
    based on bulk-density group (Table 5).

    Groups
    ------
    I  : BD ≤ 0.1
    II : 0.1 < BD ≤ 0.2
    """
    if BD <= 0.1:
        # Group I – MVG
        return MVGParameters(theta_s=0.92, alpha=0.133, n=1.28, Ks=math.nan, tau=math.nan)
    else:
        # Group II – MVG
        return MVGParameters(theta_s=0.88, alpha=0.029, n=1.22, Ks=1.95, tau=0.50)


def mvg_grouped_sedge_ptf(BD: float) -> MVGParameters:
    """PTF-derived grouped parameters for Sedge peat (Table 5)."""
    if BD <= 0.1:
        # Group I – PTF
        return MVGParameters(theta_s=0.92, alpha=0.099, n=1.30, Ks=math.nan, tau=math.nan)
    else:
        # Group II – PTF
        return MVGParameters(theta_s=0.88, alpha=0.038, n=1.21, Ks=0.91, tau=0.50)


def mvg_grouped_all(BD: float) -> MVGParameters:
    """
    Return representative **MVG-optimised** parameters for **all peat types**
    with BD > 0.2 g cm⁻³ (Table 5).
    """
    return MVGParameters(theta_s=0.77, alpha=0.014, n=1.16, Ks=0.23, tau=0.50)


def mvg_grouped_all_ptf(BD: float) -> MVGParameters:
    """PTF-derived grouped parameters for all peat types, BD > 0.2 (Table 5)."""
    return MVGParameters(theta_s=0.77, alpha=0.016, n=1.19, Ks=0.05, tau=0.50)


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

def get_mvg_parameters(
    BD: float,
    depth: float = 0.0,
    OM: Optional[float] = None,
    peat_type: Optional[str] = None,
    method: str = "ptf",
) -> MVGParameters:
    """
    High-level convenience function to obtain MVG parameters.

    Parameters
    ----------
    BD : float
        Bulk density (g cm⁻³).
    depth : float, optional
        Sample depth (cm). Default 0.
    OM : float or None, optional
        Organic matter content (wt%).
    peat_type : str or None, optional
        ``"sphagnum"``, ``"woody"``, ``"sedge"``, or ``None`` for general.
    method : str, optional
        ``"ptf"`` – use the regression equations from Table 4 (default).
        ``"grouped_mvg"`` – use the MVG-optimised grouped values from Table 5.
        ``"grouped_ptf"`` – use the PTF-derived grouped values from Table 5.

    Returns
    -------
    MVGParameters
    """
    method = method.strip().lower()

    if method == "ptf":
        return ptf_all_data(BD, depth=depth, OM=OM, peat_type=peat_type)

    # --- grouped look-ups (Table 5) ---
    ptype = (peat_type or "").strip().lower()

    if method == "grouped_mvg":
        if ptype == "sphagnum":
            return mvg_grouped_sphagnum(BD)
        elif ptype == "woody":
            return mvg_grouped_woody(BD)
        elif ptype == "sedge":
            return mvg_grouped_sedge(BD)
        else:
            return mvg_grouped_all(BD)

    elif method == "grouped_ptf":
        if ptype == "sphagnum":
            return mvg_grouped_sphagnum_ptf(BD)
        elif ptype == "woody":
            return mvg_grouped_woody_ptf(BD)
        elif ptype == "sedge":
            return mvg_grouped_sedge_ptf(BD)
        else:
            return mvg_grouped_all_ptf(BD)

    else:
        raise ValueError(
            f"Unknown method '{method}'; "
            "expected 'ptf', 'grouped_mvg', or 'grouped_ptf'."
        )


# ---------------------------------------------------------------------------
# Van Genuchten water retention & hydraulic conductivity curves
# ---------------------------------------------------------------------------

def _require_numpy() -> None:
    """Raise an informative error if NumPy is not installed."""
    if np is None:
        raise ImportError(
            "NumPy is required for water-retention and conductivity "
            "functions.  Install it with:  pip install numpy"
        )


def vg_water_retention(
    h: ArrayLike,
    params: MVGParameters,
    theta_r: Optional[float] = None,
) -> np.ndarray:
    """
    Compute the **van Genuchten water retention curve** θ(h).

    .. math::

        \\theta(h) = \\theta_r
            + \\frac{\\theta_s - \\theta_r}{[1 + (\\alpha |h|)^n]^m}

    where *m = 1 − 1/n*.

    Parameters
    ----------
    h : array_like
        Pressure head (cm).  Negative values denote unsaturated conditions
        (suction); positive values are clamped to saturation.
    params : MVGParameters
        MVG parameter set (must contain θs, α, n and, optionally, θr).
    theta_r : float or None, optional
        Override the residual water content.  When ``None`` the value stored
        in *params* is used (default 0.0 for peat soils).

    Returns
    -------
    np.ndarray
        Volumetric water content θ (cm³ cm⁻³) at each pressure head.
    """
    _require_numpy()
    h = np.asarray(h, dtype=float)
    tr = theta_r if theta_r is not None else params.theta_r
    ts = params.theta_s
    alpha = params.alpha
    n = params.n
    m = 1.0 - 1.0 / n

    # Use absolute value of h; at h >= 0 the soil is saturated
    abs_h = np.abs(h)
    theta = tr + (ts - tr) / (1.0 + (alpha * abs_h) ** n) ** m

    # Saturated where h >= 0
    theta = np.where(h >= 0, ts, theta)
    return theta


def vg_effective_saturation(
    h: ArrayLike,
    params: MVGParameters,
    theta_r: Optional[float] = None,
) -> np.ndarray:
    """
    Effective saturation Se(h) ∈ [0, 1].

    .. math::

        S_e = \\frac{\\theta - \\theta_r}{\\theta_s - \\theta_r}
            = \\frac{1}{[1 + (\\alpha |h|)^n]^m}
    """
    _require_numpy()
    h = np.asarray(h, dtype=float)
    tr = theta_r if theta_r is not None else params.theta_r
    alpha = params.alpha
    n = params.n
    m = 1.0 - 1.0 / n

    abs_h = np.abs(h)
    Se = 1.0 / (1.0 + (alpha * abs_h) ** n) ** m
    Se = np.where(h >= 0, 1.0, Se)
    return Se


def vg_hydraulic_conductivity(
    h: ArrayLike,
    params: MVGParameters,
    theta_r: Optional[float] = None,
) -> np.ndarray:
    """
    Compute the **Mualem–van Genuchten unsaturated hydraulic conductivity**
    K(h).

    .. math::

        K(h) = K_s \\, S_e^{\\tau}
               \\left[1 - \\left(1 - S_e^{1/m}\\right)^m \\right]^2

    Parameters
    ----------
    h : array_like
        Pressure head (cm), negative = unsaturated.
    params : MVGParameters
        MVG parameter set.
    theta_r : float or None, optional
        Override residual water content.

    Returns
    -------
    np.ndarray
        Hydraulic conductivity K (cm h⁻¹) at each pressure head.
    """
    _require_numpy()
    Se = vg_effective_saturation(h, params, theta_r=theta_r)
    m = 1.0 - 1.0 / params.n
    Ks = params.Ks
    tau = params.tau

    K = Ks * Se ** tau * (1.0 - (1.0 - Se ** (1.0 / m)) ** m) ** 2
    return np.where(np.asarray(h, dtype=float) >= 0, Ks, K)


def plot_swrc(
    params_dict: dict[str, MVGParameters],
    h_range: tuple[float, float] = (1e-2, 1e5),
    n_points: int = 500,
    theta_r: Optional[float] = None,
    ax: Optional[object] = None,
    log_x: bool = True,
) -> object:
    """
    Plot **soil water retention curves** (SWRCs) for one or more MVG
    parameter sets.

    Parameters
    ----------
    params_dict : dict[str, MVGParameters]
        Mapping of *label* → *MVGParameters*.  Each entry becomes one curve.
    h_range : tuple[float, float], optional
        Min and max |h| (cm) for the x-axis.  Default ``(0.01, 100000)``.
    n_points : int, optional
        Number of points along the h axis.  Default 500.
    theta_r : float or None, optional
        Override residual water content for all curves.
    ax : matplotlib Axes or None, optional
        Axes to draw on.  If ``None`` a new figure is created.
    log_x : bool, optional
        Use a log scale for the pressure-head axis (default ``True``).

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    h = np.logspace(np.log10(h_range[0]), np.log10(h_range[1]), n_points)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    for label, params in params_dict.items():
        theta = vg_water_retention(-h, params, theta_r=theta_r)
        ax.plot(h, theta, linewidth=2, label=label)

    ax.set_xlabel("|Pressure head|  (cm)")
    ax.set_ylabel(r"Volumetric water content $\theta$  (cm$^3$ cm$^{-3}$)")
    ax.set_title("Soil Water Retention Curves (van Genuchten)")
    if log_x:
        ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    return ax


def plot_k_curve(
    params_dict: dict[str, MVGParameters],
    h_range: tuple[float, float] = (1e-2, 1e5),
    n_points: int = 500,
    theta_r: Optional[float] = None,
    ax: Optional[object] = None,
) -> object:
    """
    Plot **unsaturated hydraulic conductivity** K(h) for one or more MVG
    parameter sets.

    Parameters (same as :func:`plot_swrc`).

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    h = np.logspace(np.log10(h_range[0]), np.log10(h_range[1]), n_points)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    for label, params in params_dict.items():
        if math.isnan(params.Ks):
            continue
        K = vg_hydraulic_conductivity(-h, params, theta_r=theta_r)
        ax.plot(h, K, linewidth=2, label=label)

    ax.set_xlabel("|Pressure head|  (cm)")
    ax.set_ylabel(r"Hydraulic conductivity $K$  (cm h$^{-1}$)")
    ax.set_title("Unsaturated Hydraulic Conductivity (Mualem–van Genuchten)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    return ax


def plot_mvg(
    theta_s: float,
    alpha: float,
    n: float,
    Ks: float = math.nan,
    tau: float = 0.5,
    theta_r: float = 0.0,
    *,
    label: str = "Custom MVG",
    h_range: tuple[float, float] = (1e-2, 1e5),
    n_points: int = 500,
    show_k: bool = True,
    figsize: tuple[float, float] = (14, 5),
    colors: Optional[tuple[str, str]] = None,
    ax_swrc: Optional[object] = None,
    ax_k: Optional[object] = None,
) -> object:
    """
    Plot a **customised MVG model** from individual parameter values.

    Creates a side-by-side figure with the soil-water retention curve (left)
    and the unsaturated hydraulic conductivity curve (right).  All six MVG
    parameters can be specified directly – no need to build a
    ``MVGParameters`` object first.

    Parameters
    ----------
    theta_s : float
        Saturated water content (cm³ cm⁻³).
    alpha : float
        van Genuchten α (cm⁻¹).
    n : float
        van Genuchten n (dimensionless, > 1).
    Ks : float, optional
        Saturated hydraulic conductivity (cm h⁻¹).  If ``NaN`` (default),
        the K(h) panel is omitted.
    tau : float, optional
        Mualem tortuosity (default 0.5).
    theta_r : float, optional
        Residual water content (default 0.0).
    label : str, optional
        Legend label for the curve (default ``"Custom MVG"``).
    h_range : tuple[float, float], optional
        Min and max |h| in cm (default ``(0.01, 100000)``).
    n_points : int, optional
        Number of evaluation points (default 500).
    show_k : bool, optional
        Whether to include the K(h) panel (default ``True``).
        Ignored when *Ks* is NaN.
    figsize : tuple[float, float], optional
        Figure size in inches (default ``(14, 5)``).
    colors : tuple[str, str] or None, optional
        ``(swrc_color, k_color)``.  If ``None``, matplotlib defaults are used.
    ax_swrc : matplotlib Axes or None, optional
        Pre-existing axes for the SWRC panel.
    ax_k : matplotlib Axes or None, optional
        Pre-existing axes for the K(h) panel.

    Returns
    -------
    tuple[matplotlib.axes.Axes, matplotlib.axes.Axes | None]
        The SWRC axes and (if plotted) the K(h) axes.

    Examples
    --------
    >>> plot_mvg(theta_s=0.93, alpha=1.0, n=1.3, Ks=500, tau=-4.0)
    >>> plot_mvg(theta_s=0.88, alpha=0.05, n=1.2, label="Dense peat")
    """
    import matplotlib.pyplot as plt

    _require_numpy()
    params = MVGParameters(
        theta_s=theta_s, alpha=alpha, n=n,
        Ks=Ks, tau=tau, theta_r=theta_r,
    )
    h = np.logspace(np.log10(h_range[0]), np.log10(h_range[1]), n_points)

    plot_k = show_k and not math.isnan(Ks)

    # ---- create figure / axes ----
    if ax_swrc is None:
        ncols = 2 if plot_k else 1
        fig, axes = plt.subplots(1, ncols, figsize=figsize)
        if ncols == 1:
            ax_swrc = axes
            ax_k_out = None
        else:
            ax_swrc, ax_k = axes
            ax_k_out = ax_k
    else:
        fig = ax_swrc.figure
        ax_k_out = ax_k

    swrc_color = colors[0] if colors else None
    k_color = colors[1] if colors and len(colors) > 1 else swrc_color

    # ---- SWRC panel ----
    theta = vg_water_retention(-h, params)
    ax_swrc.plot(h, theta, linewidth=2, label=label, color=swrc_color)
    ax_swrc.set_xscale("log")
    ax_swrc.set_xlabel("|Pressure head|  (cm)")
    ax_swrc.set_ylabel(r"$\theta$  (cm$^3$ cm$^{-3}$)")
    ax_swrc.set_title("Soil Water Retention Curve")
    ax_swrc.set_ylim(bottom=0)
    ax_swrc.legend(fontsize=9)
    ax_swrc.grid(True, which="both", alpha=0.3)

    # Annotate parameter values
    m = 1.0 - 1.0 / n
    text = (
        f"θs = {theta_s:.3f}\n"
        f"θr = {theta_r:.3f}\n"
        f"α  = {alpha:.4f}\n"
        f"n  = {n:.4f}  (m = {m:.4f})"
    )
    ax_swrc.text(
        0.97, 0.97, text, transform=ax_swrc.transAxes,
        fontsize=8, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5),
        fontfamily="monospace",
    )

    # ---- K(h) panel ----
    if plot_k and ax_k is not None:
        K = vg_hydraulic_conductivity(-h, params)
        ax_k.plot(h, K, linewidth=2, label=label, color=k_color)
        ax_k.set_xscale("log")
        ax_k.set_yscale("log")
        ax_k.set_xlabel("|Pressure head|  (cm)")
        ax_k.set_ylabel(r"$K$  (cm h$^{-1}$)")
        ax_k.set_title("Hydraulic Conductivity")
        ax_k.legend(fontsize=9)
        ax_k.grid(True, which="both", alpha=0.3)

        k_text = f"Ks = {Ks:.4f}\nτ  = {tau:.4f}"
        ax_k.text(
            0.97, 0.97, k_text, transform=ax_k.transAxes,
            fontsize=8, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5),
            fontfamily="monospace",
        )
        ax_k_out = ax_k

    plt.tight_layout()
    return ax_swrc, ax_k_out


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Table 4 PTF examples ===")
    print("Sphagnum (BD=0.05, OM=95, depth=10):")
    print(f"  {ptf_sphagnum(0.05, 95, 10)}")

    print("Woody (BD=0.10, depth=20, OM=90):")
    print(f"  {ptf_woody(0.10, 20, 90)}")

    print("Sedge (BD=0.15, depth=30):")
    print(f"  {ptf_sedge(0.15, 30)}")

    print("All types high BD (BD=0.35):")
    print(f"  {ptf_all_types_high_bd(0.35)}")

    print("All data general (BD=0.10, depth=15):")
    print(f"  {ptf_all_data(0.10, depth=15)}")

    print()
    print("=== Table 5 grouped examples ===")
    print("Sphagnum grouped MVG (BD=0.03):")
    print(f"  {mvg_grouped_sphagnum(0.03)}")

    print("Woody grouped PTF (BD=0.08):")
    print(f"  {mvg_grouped_woody_ptf(0.08)}")

    print()
    print("=== Convenience dispatcher ===")
    params = get_mvg_parameters(BD=0.12, depth=25, OM=92, peat_type="sphagnum", method="ptf")
    print(f"  {params}")
