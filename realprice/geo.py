"""Geographic helpers.

* Conversion between the Israeli grid (ITM, EPSG:2039, metres) and GPS coordinates (WGS84, degrees).
* A fast "nearest point" / "how many points within a radius" search.

Why write the projection by hand? The formulas (Krueger series) are accurate to about 1 mm over Israel, and
this way the project needs no heavy GIS library (pyproj / GDAL). It is vectorised with numpy, so it converts
millions of points quickly.
"""
import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- constants of the GRS80 ellipsoid and the ITM projection
A = 6378137.0                  # equatorial radius (m)
F = 1 / 298.257222101          # flattening
N = F / (2 - F)                # "third flattening", the small number the series are expanded in
A_HAT = A / (1 + N) * (1 + N ** 2 / 4 + N ** 4 / 64)   # rectifying radius
E = 2 * np.sqrt(N) / (1 + N)   # eccentricity
# Series coefficients: ALPHA is used for lat/lon -> ITM, BETA and DELTA for ITM -> lat/lon.
ALPHA = (N / 2 - 2 * N ** 2 / 3 + 5 * N ** 3 / 16, 13 * N ** 2 / 48 - 3 * N ** 3 / 5, 61 * N ** 3 / 240)
BETA = (N / 2 - 2 * N ** 2 / 3 + 37 * N ** 3 / 96, N ** 2 / 48 + N ** 3 / 15, 17 * N ** 3 / 480)
DELTA = (2 * N - 2 * N ** 2 / 3 - 2 * N ** 3, 7 * N ** 2 / 3 - 8 * N ** 3 / 5, 56 * N ** 3 / 15)
# Origin of the ITM grid.
LAT0, LON0 = np.radians(31.7343936111), np.radians(35.2045169444)
SCALE, FALSE_EASTING, FALSE_NORTHING = 1.0000067, 219529.584, 626907.390

# Israel in ITM metres (a generous box, includes the West Bank and the Golan). Anything outside is garbage.
ITM_BOUNDS = {"x": (80_000, 320_000), "y": (350_000, 810_000)}


def _apply_series(coefficients, xi, eta, sign):
    """The correction terms shared by both directions: sum of c_j * sin(2j*xi) * cosh(2j*eta) and its partner."""
    d_xi = sum(c * np.sin(2 * j * xi) * np.cosh(2 * j * eta) for j, c in enumerate(coefficients, 1))
    d_eta = sum(c * np.cos(2 * j * xi) * np.sinh(2 * j * eta) for j, c in enumerate(coefficients, 1))
    return xi + sign * d_xi, eta + sign * d_eta


def _forward(lat, delta_lon):
    """Radians (latitude, longitude relative to the origin) -> the projection's own xi / eta coordinates."""
    s = np.sin(lat)
    t = np.sinh(np.arctanh(s) - E * np.arctanh(E * s))
    xi = np.arctan2(t, np.cos(delta_lon))
    eta = np.arctanh(np.sin(delta_lon) / np.sqrt(1 + t * t))
    return _apply_series(ALPHA, xi, eta, +1)


XI_AT_ORIGIN = float(_forward(np.float64(LAT0), np.float64(0.0))[0])


def wgs84_to_itm(lat, lon):
    """Degrees -> ITM metres. Returns (x = easting, y = northing)."""
    xi, eta = _forward(np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float)) - LON0)
    return FALSE_EASTING + SCALE * A_HAT * eta, FALSE_NORTHING + SCALE * A_HAT * (xi - XI_AT_ORIGIN)


def itm_to_wgs84(x, y):
    """ITM metres -> (lat, lon) in degrees."""
    xi = (np.asarray(y, float) - FALSE_NORTHING) / (SCALE * A_HAT) + XI_AT_ORIGIN
    eta = (np.asarray(x, float) - FALSE_EASTING) / (SCALE * A_HAT)
    xi, eta = _apply_series(BETA, xi, eta, -1)
    chi = np.arcsin(np.sin(xi) / np.cosh(eta))
    lat = chi + sum(d * np.sin(2 * j * chi) for j, d in enumerate(DELTA, 1))
    lon = LON0 + np.arctan2(np.sinh(eta), np.cos(xi))
    return np.degrees(lat), np.degrees(lon)


def looks_like_degrees(a, b) -> bool:
    """True if both arrays look like lat / lon in degrees (small numbers) rather than ITM metres (large numbers)."""
    return bool(np.nanmax(np.abs(a)) < 200 and np.nanmax(np.abs(b)) < 200)


def inside_itm_bounds(x, y):
    (x_min, x_max), (y_min, y_max) = ITM_BOUNDS["x"], ITM_BOUNDS["y"]
    return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)


def ring_centroid(ring):
    """Centre of a polygon ring -> (x, y, area). The centre is area-weighted. Returns area 0 for a degenerate ring.

    The coordinates are shifted so that the first point is the origin; with big ITM numbers the
    arithmetic would otherwise lose precision.
    """
    ring = np.asarray(ring, float)
    origin = ring[0]
    x, y = ring[:, 0] - origin[0], ring[:, 1] - origin[1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = cross.sum() / 2
    if abs(area) < 1e-9:
        return x.mean() + origin[0], y.mean() + origin[1], 0.0
    cx = ((x + x_next) * cross).sum() / (6 * area)
    cy = ((y + y_next) * cross).sum() / (6 * area)
    return cx + origin[0], cy + origin[1], abs(area)


class PointIndex:
    """A KD-tree over points given in metres. Finds the nearest point or counts the points within a radius
    for many query points at once, much faster than comparing every pair of points."""

    def __init__(self, x, y):
        self.tree = cKDTree(np.column_stack([x, y]))

    def nearest(self, x, y):
        """Distance (m) and index of the nearest indexed point, for every query point."""
        distance, index = self.tree.query(np.column_stack([x, y]), workers=-1)
        return distance, index

    def count_within(self, x, y, radius):
        """How many indexed points are within `radius` metres of every query point."""
        return self.tree.query_ball_point(np.column_stack([x, y]), radius, return_length=True, workers=-1)
