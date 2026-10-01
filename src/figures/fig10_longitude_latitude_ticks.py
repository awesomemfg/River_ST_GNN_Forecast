# Revision 14 (Farid, 2026-09-28): label the Figure 10 map in longitude and latitude instead of state-plane feet.
# This block replaces the million-feet tick formatter of the native recipe
# (20260826_Final_V2/figure_revision_scripts/recipe_spatial_skill_map.py). The basemap stays in its
# native projection (make_basemap), and the map extent, markers, and colors do not change. Only the tick positions and
# labels change: x ticks sit at the projected x of round longitudes along the middle latitude of the map, and y ticks
# at the projected y of round latitudes along the middle longitude.
from pyproj import Transformer
from _mapbase import HECRAS_CRS, LONLAT

map_to_lonlat = Transformer.from_crs(HECRAS_CRS, LONLAT, always_xy=True)
lonlat_to_map = Transformer.from_crs(LONLAT, HECRAS_CRS, always_xy=True)
x_low, x_high = map_axis.get_xlim()
y_low, y_high = map_axis.get_ylim()
x_middle = 0.5 * (x_low + x_high)
y_middle = 0.5 * (y_low + y_high)
lon_low, lat_middle = map_to_lonlat.transform(x_low, y_middle)
lon_high, lat_unused = map_to_lonlat.transform(x_high, y_middle)
lon_middle, lat_low = map_to_lonlat.transform(x_middle, y_low)
lon_unused, lat_high = map_to_lonlat.transform(x_middle, y_high)


def round_degree_ticks(low, high):
    """Round degree values inside (low, high), using the finest step that gives at most six ticks."""
    for step in (0.05, 0.1, 0.2, 0.25, 0.5):
        values = np.arange(np.ceil(low / step) * step, high, step)
        if len(values) <= 6:
            return values, step
    return values, step


lon_values, lon_step = round_degree_ticks(lon_low, lon_high)
lat_values, lat_step = round_degree_ticks(lat_low, lat_high)
x_ticks = [lonlat_to_map.transform(lon, lat_middle)[0] for lon in lon_values]
y_ticks = [lonlat_to_map.transform(lon_middle, lat)[1] for lat in lat_values]
lon_decimals = 2 if lon_step < 0.1 else 1
lat_decimals = 2 if lat_step < 0.1 else 1
map_axis.set_xticks(x_ticks)
map_axis.set_xticklabels([f"{abs(lon):.{lon_decimals}f}°W" for lon in lon_values])
map_axis.set_yticks(y_ticks)
map_axis.set_yticklabels([f"{lat:.{lat_decimals}f}°N" for lat in lat_values])
map_axis.set_xlim(x_low, x_high)
map_axis.set_ylim(y_low, y_high)
map_axis.set_xlabel("Longitude", fontsize=12.5)
map_axis.set_ylabel("Latitude", fontsize=12.5)
