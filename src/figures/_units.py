"""Unit conversion — the operational data is stored in imperial units (river stage and RMSE in feet,
rainfall in inches), but this paper presents everything in metric (SI / HESS convention).

Exact factors (no rounding in the factor itself):
  1 ft (international)        = 0.3048 m exactly        -> stage, RMSE, rise, elevation
  1 in                       = 25.4 mm exactly          -> rainfall
  1 US survey foot           = 1200/3937 m              -> NAD83 LA-South StatePlane coordinates
                             ≈ 0.304800609601 m         (differs from the int. foot by ~2 ppm)

Conversions are applied at the DISPLAY / loader layer; the raw stored data and the routing/ranking
thresholds (which are defined in feet) are left untouched so the science is unchanged.
"""
from matplotlib.ticker import FuncFormatter

FT2M = 0.3048          # feet -> metres (river stage, RMSE, rise, DEM elevation)
IN2MM = 25.4           # inches -> millimetres (rainfall)
FTUS2M = 1200.0 / 3937.0   # US survey foot -> metres (StatePlane easting/northing)


def m(x):
    """Feet -> metres. Accepts a scalar, numpy array, or pandas Series."""
    return x * FT2M


def mm(x):
    """Inches -> millimetres."""
    return x * IN2MM


def label_axes_m(ax, which="both"):
    """Relabel a StatePlane axis whose tick values are in US survey feet so it DISPLAYS metres.

    The underlying geometry stays in StatePlane feet (no reprojection); only the tick labels are
    reformatted (value * FTUS2M). Call this AFTER any ax.ticklabel_format(...) so it is not
    overridden by the ScalarFormatter that call installs.
    """
    fmt = FuncFormatter(lambda v, pos: f"{v * FTUS2M:,.0f}")
    if which in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if which in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)
