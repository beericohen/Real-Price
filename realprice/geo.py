"""Geographic helpers: Israeli Transverse Mercator (ITM, EPSG:2039) <-> WGS84 and fast nearest-point search.

ITM is implemented here with Krueger's series (accurate to ~1 mm over Israel), vectorised with numpy,
so no heavy GIS dependency (pyproj/GDAL) is needed.
"""
import numpy as np
from scipy.spatial import cKDTree

# GRS80 ellipsoid + ITM projection parameters
_A, _F = 6378137.0, 1 / 298.257222101
_N = _F / (2 - _F)
_AA = _A / (1 + _N) * (1 + _N ** 2 / 4 + _N ** 4 / 64)
_ALPHA = (_N / 2 - 2 * _N ** 2 / 3 + 5 * _N ** 3 / 16, 13 * _N ** 2 / 48 - 3 * _N ** 3 / 5, 61 * _N ** 3 / 240)
_BETA = (_N / 2 - 2 * _N ** 2 / 3 + 37 * _N ** 3 / 96, _N ** 2 / 48 + _N ** 3 / 15, 17 * _N ** 3 / 480)
_DELTA = (2 * _N - 2 * _N ** 2 / 3 - 2 * _N ** 3, 7 * _N ** 2 / 3 - 8 * _N ** 3 / 5, 56 * _N ** 3 / 15)
_E = 2 * np.sqrt(_N) / (1 + _N)
LAT0, LON0 = np.radians(31.7343936111), np.radians(35.2045169444)
K0, FALSE_E, FALSE_N = 1.0000067, 219529.584, 626907.390

# Israel in ITM metres (generous box, includes the West Bank / Golan) - used to reject garbage coordinates
ITM_BOUNDS = {"x": (80_000, 320_000), "y": (350_000, 810_000)}


def _series(coeffs, xi, eta, sign):
    """Sum_j c_j * sin/cos(2j*xi) * cosh/sinh(2j*eta) terms shared by the forward and inverse formulas."""
    d_xi = sum(c * np.sin(2 * j * xi) * np.cosh(2 * j * eta) for j, c in enumerate(coeffs, 1))
    d_eta = sum(c * np.cos(2 * j * xi) * np.sinh(2 * j * eta) for j, c in enumerate(coeffs, 1))
    return xi + sign * d_xi, eta + sign * d_eta


def _forward_xi_eta(lat, dlon):
    s = np.sin(lat)
    t = np.sinh(np.arctanh(s) - _E * np.arctanh(_E * s))
    xi1 = np.arctan2(t, np.cos(dlon))
    eta1 = np.arctanh(np.sin(dlon) / np.sqrt(1 + t * t))
    return _series(_ALPHA, xi1, eta1, +1)


_XI0 = float(_forward_xi_eta(np.float64(LAT0), np.float64(0.0))[0])


def wgs84_to_itm(lat, lon):
    """Degrees -> ITM metres (x = easting, y = northing)."""
    xi, eta = _forward_xi_eta(np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float)) - LON0)
    return FALSE_E + K0 * _AA * eta, FALSE_N + K0 * _AA * (xi - _XI0)


def itm_to_wgs84(x, y):
    """ITM metres -> (lat, lon) in degrees."""
    xi = (np.asarray(y, float) - FALSE_N) / (K0 * _AA) + _XI0
    eta = (np.asarray(x, float) - FALSE_E) / (K0 * _AA)
    xi1, eta1 = _series(_BETA, xi, eta, -1)
    chi = np.arcsin(np.sin(xi1) / np.cosh(eta1))
    lat = chi + sum(d * np.sin(2 * j * chi) for j, d in enumerate(_DELTA, 1))
    lon = LON0 + np.arctan2(np.sinh(eta1), np.cos(xi1))
    return np.degrees(lat), np.degrees(lon)


def looks_like_degrees(a, b) -> bool:
    """True when both arrays look like lat/lon degrees instead of ITM metres."""
    return bool(np.nanmax(np.abs(a)) < 200 and np.nanmax(np.abs(b)) < 200)


def inside_itm_bounds(x, y):
    (x0, x1), (y0, y1) = ITM_BOUNDS["x"], ITM_BOUNDS["y"]
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def ring_centroid(ring) -> tuple[float, float, float]:
    """Area-weighted centroid of a polygon ring -> (cx, cy, |area|). Coordinates are shifted for precision."""
    ring = np.asarray(ring, float)
    origin = ring[0]
    x, y = ring[:, 0] - origin[0], ring[:, 1] - origin[1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2
    if abs(area) < 1e-9:
        return x.mean() + origin[0], y.mean() + origin[1], 0.0
    cx = ((x + x1) * cross).sum() / (6 * area)
    cy = ((y + y1) * cross).sum() / (6 * area)
    return cx + origin[0], cy + origin[1], abs(area)


class PointIndex:
    """KD-tree over projected (metre) points: O(log n) nearest-neighbour and radius-count queries."""

    def __init__(self, x, y):
        self.tree = cKDTree(np.column_stack([x, y]))

    def nearest(self, x, y):
        """Distance (m) and index of the nearest indexed point for every query point."""
        dist, idx = self.tree.query(np.column_stack([x, y]), workers=-1)
        return dist, idx

    def count_within(self, x, y, radius):
        """How many indexed points lie within `radius` metres of every query point."""
        return self.tree.query_ball_point(np.column_stack([x, y]), radius, return_length=True, workers=-1)
