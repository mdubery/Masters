"""Mass-loss rates interpolated from the Aiolos hydrodynamic grid.

Each table is one planet mass and equilibrium temperature. Inside the
tabulated range the rate is multilinear interpolation in
log(mass), log(temperature), log(transit radius) and log(XUV luminosity).
Outside that range the rate is continued from the edge of the table:
energy-limited at radii below R_trans, and an isothermal Parker wind at
and above R_trans (Misener et al. 2026, equation 11).

The Parker sonic radius is R_s = GM / (2 c_s^2), with
c_s^2 = k_B T_eq / (mu m_H).
"""
import glob
import os
import re

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIRECTORY = os.path.join(_MODULE_DIR, "Aiolos mass loss rates", "V2")

# Same constants as the rest of the model, repeated here so this file can
# be read on its own. Lengths below are used in centimetres and metres.
_R_EARTH_CM = 6.378e8
_R_EARTH_M = 6.378e6
_G_SI = 6.674e-11
_M_EARTH_KG = 5.972e24
_K_B = 1.381e-23
_MU = 2.35
_M_H = 1.67e-27
_LN10 = np.log(10.0)


def _R_trans_earth(M_earth, Teq_k):
    """Transition radius of Misener et al. (2026), equation 11, in Earth radii.

    Below this radius the out-of-table continuation is energy-limited.
    At and above it, the continuation is a Parker wind.
    """
    return 4.2 * (M_earth / 5.0) ** 0.85 * (Teq_k / 1000.0) ** (-1.05)


def _R_sonic_earth(mass_earth, teq_k):
    """Isothermal sonic radius GM / (2 c_s^2), in Earth radii.

    c_s^2 = k_B T_eq / (mu m_H).
    """
    c_s_squared = _K_B * teq_k / (_MU * _M_H)
    return _G_SI * (mass_earth * _M_EARTH_KG) / (2.0 * c_s_squared) / _R_EARTH_M


def _slice_extrap(log_anchor, log_r_anchor, log_l_anchor, log_r, log_l, mass_earth, teq_k):
    """Continue one mass-temperature slice outside its tabulated radius and luminosity.

    log_anchor is log10(mass-loss rate) at the anchor point
    (log_r_anchor, log_l_anchor). Mass and temperature are held fixed, so
    they do not add an increment. The increment is energy-limited below
    R_trans and a Parker wind at and above it.
    """
    dlog_r = log_r - log_r_anchor
    dlog_l = log_l - log_l_anchor
    dlog_el = 3.0 * dlog_r + dlog_l

    radius = np.maximum(10.0 ** log_r / _R_EARTH_CM, 1e-6)
    radius_anchor = np.maximum(10.0 ** log_r_anchor / _R_EARTH_CM, 1e-6)
    rs = _R_sonic_earth(mass_earth, teq_k)
    dlog_parker = (-2.0 * rs / radius + 2.0 * rs / radius_anchor) / _LN10

    r_trans = np.maximum(_R_trans_earth(mass_earth, teq_k), 1e-12)
    x = np.log10(np.maximum(radius, 1e-12) / r_trans)
    w = np.where(x >= 0.0, 1.0, 0.0)
    return log_anchor + (1.0 - w) * dlog_el + w * dlog_parker


def _fill_rl_plane(log_r, log_l, log_mdot, log_R_ax, log_L_ax, mass_earth, teq_k):
    """Fill the radius-luminosity plane of one mass and temperature.

    Points inside the convex hull of that slice are linearly interpolated.
    Points outside it are continued from the edge of the same slice, never
    from a different planet mass.
    """
    from scipy.spatial import cKDTree

    pts = np.column_stack([np.asarray(log_r, dtype=float), np.asarray(log_l, dtype=float)])
    log_mdot = np.asarray(log_mdot, dtype=float)
    rr, ll = np.meshgrid(log_R_ax, log_L_ax, indexing="ij")
    query = np.column_stack([rr.ravel(), ll.ravel()])
    plane = np.full(query.shape[0], np.nan)

    linear = None
    if len(pts) >= 4:
        try:
            linear = LinearNDInterpolator(pts, log_mdot, fill_value=np.nan)
            plane = np.asarray(linear(query), dtype=float)
        except Exception:
            linear = None

    missing = ~np.isfinite(plane)
    if not missing.any():
        return plane.reshape(len(log_R_ax), len(log_L_ax))

    q = query[missing]
    anchor_r = np.clip(q[:, 0], log_r.min(), log_r.max())
    anchor_l = np.clip(q[:, 1], log_l.min(), log_l.max())
    anchor_val = np.full(q.shape[0], np.nan)
    if linear is not None:
        anchor_val = np.asarray(linear(np.column_stack([anchor_r, anchor_l])), dtype=float)

    still = ~np.isfinite(anchor_val)
    if still.any():
        _, idx = cKDTree(pts).query(q[still])
        anchor_val[still] = log_mdot[idx]
        anchor_r[still] = pts[idx, 0]
        anchor_l[still] = pts[idx, 1]

    plane[missing] = _slice_extrap(
        anchor_val, anchor_r, anchor_l, q[:, 0], q[:, 1], mass_earth, teq_k,
    )
    return plane.reshape(len(log_R_ax), len(log_L_ax))


def _resolve_directory(directory):
    if os.path.isdir(directory):
        return os.path.abspath(directory)
    alt = os.path.join(_MODULE_DIR, directory)
    if os.path.isdir(alt):
        return alt
    raise FileNotFoundError(
        f"Aiolos mass-loss directory not found: {directory!r}"
    )


class AiolosMassLossGrid:
    """Interpolator for the Aiolos mass-loss tables.

    A query is (planet mass in Earth masses, equilibrium temperature in
    kelvin, transit radius in centimetres, XUV luminosity in erg/s). The
    returned rate is in g/s.

    Each simulated (mass, temperature) pair is placed on a shared grid in
    radius and XUV luminosity. A grid node outside that pair's own models
    is filled by the energy-limited or Parker continuation. A
    (mass, temperature) pair that was never simulated is filled from the
    pairs that were. Later queries use that regular grid.

    Rows with a non-finite or non-positive luminosity, radius, or rate are
    dropped. Only CSV files in the top directory are read.
    """

    FILE_RE = re.compile(r"M(?P<mass>\d+)_Teq(?P<teq>\d+)")

    # Resolution of the pre-built regular grid along the R and Luv axes.
    _N_RADIUS = 80
    _N_LUV    = 80

    def __init__(self, directory=DEFAULT_DIRECTORY):
        self.directory = _resolve_directory(directory)
        self.points, self.values, self.load_report = self._load_points(self.directory)

        # Mass and temperature stay on the simulated values. Radius and
        # luminosity are resampled onto one shared axis for every slice.
        self._log_mass_ax = np.unique(self.points[:, 0])
        self._log_teq_ax  = np.unique(self.points[:, 1])
        self._log_R_ax   = np.linspace(self.points[:, 2].min(),
                                        self.points[:, 2].max(), self._N_RADIUS)
        self._log_Luv_ax = np.linspace(self.points[:, 3].min(),
                                        self.points[:, 3].max(), self._N_LUV)

        nM = len(self._log_mass_ax)
        nT = len(self._log_teq_ax)
        nR = self._N_RADIUS
        nL = self._N_LUV
        grid_vals = np.full((nM, nT, nR, nL), np.nan)

        # One plane per simulated mass and temperature. Do not mix masses.
        for log_mass in self._log_mass_ax:
            for log_teq in self._log_teq_ax:
                sel = (
                    np.isclose(self.points[:, 0], log_mass)
                    & np.isclose(self.points[:, 1], log_teq)
                )
                if not np.any(sel):
                    continue
                iM = int(np.where(self._log_mass_ax == log_mass)[0][0])
                iT = int(np.where(self._log_teq_ax == log_teq)[0][0])
                grid_vals[iM, iT] = _fill_rl_plane(
                    self.points[sel, 2], self.points[sel, 3], self.values[sel],
                    self._log_R_ax, self._log_Luv_ax,
                    float(10.0 ** log_mass), float(10.0 ** log_teq),
                )

        # Pairs that were never simulated: interpolate in log mass and log
        # temperature from the slices that exist.
        present = np.isfinite(grid_vals[:, :, 0, 0])
        if not present.all():
            ii, jj = np.where(present)
            coords = np.column_stack([self._log_mass_ax[ii], self._log_teq_ax[jj]])
            values = grid_vals[ii, jj].reshape(len(ii), -1)
            try:
                linear_mt = LinearNDInterpolator(coords, values, fill_value=np.nan)
            except Exception:
                linear_mt = None
            nearest_mt = NearestNDInterpolator(coords, values)
            for iM, iT in zip(*np.where(~present)):
                q = np.array([[self._log_mass_ax[iM], self._log_teq_ax[iT]]])
                filled = None if linear_mt is None else np.asarray(linear_mt(q), dtype=float).ravel()
                if filled is None or not np.isfinite(filled).all():
                    filled = np.asarray(nearest_mt(q), dtype=float).ravel()
                grid_vals[iM, iT] = filled.reshape(nR, nL)

        # The regular-grid interpolant continues the last cell outside the
        # axes. _log_mdot replaces that continuation with the physical one.
        self._rgi = RegularGridInterpolator(
            (self._log_mass_ax, self._log_teq_ax, self._log_R_ax, self._log_Luv_ax),
            grid_vals,
            method='linear',
            bounds_error=False,
            fill_value=None,
        )

        self.mass_range      = 10 ** self._log_mass_ax[[0, -1]]
        self.teq_range       = 10 ** self._log_teq_ax[[0, -1]]
        self.radius_range_cm = 10 ** np.array([self._log_R_ax[0],   self._log_R_ax[-1]])
        self.luv_range       = 10 ** np.array([self._log_Luv_ax[0], self._log_Luv_ax[-1]])

    @classmethod
    def _load_points(cls, directory):
        """Read the CSV tables, dropping non-finite and non-positive entries.

        A filename such as M5000_Teq1000 means 5 Earth masses and 1000 K.
        The integer after M is the mass in units of 0.001 Earth masses.
        """
        points_list = []
        values_list = []
        n_read = 0
        n_nan = 0
        n_nonpos = 0
        n_kept = 0
        slices = []

        for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
            match = cls.FILE_RE.search(os.path.basename(path))
            if match is None:
                continue

            mass_earth = float(match.group("mass")) / 1000.0
            teq        = float(match.group("teq"))

            arr = np.genfromtxt(path, delimiter=",")
            arr = np.atleast_2d(arr)
            if arr.shape[0] < 5:
                continue

            luv               = np.asarray(arr[1], dtype=float)
            transit_radius_cm = np.asarray(arr[2], dtype=float)
            mdot_g_s          = np.asarray(arr[4], dtype=float)
            n_read += mdot_g_s.size

            finite = (
                np.isfinite(luv)
                & np.isfinite(transit_radius_cm)
                & np.isfinite(mdot_g_s)
            )
            n_nan += int((~finite).sum())
            positive = (luv > 0.0) & (transit_radius_cm > 0.0) & (mdot_g_s > 0.0)
            n_nonpos += int((finite & ~positive).sum())
            mask = finite & positive
            if not mask.any():
                continue

            n = int(mask.sum())
            n_kept += n
            pts = np.empty((n, 4))
            pts[:, 0] = np.log10(mass_earth)
            pts[:, 1] = np.log10(teq)
            pts[:, 2] = np.log10(transit_radius_cm[mask])
            pts[:, 3] = np.log10(luv[mask])

            points_list.append(pts)
            values_list.append(np.log10(mdot_g_s[mask]))
            slices.append((mass_earth, teq, n))

        if not points_list:
            raise FileNotFoundError(
                f"No valid Aiolos grid points found in {directory!r}"
            )

        report = {
            "n_read": n_read,
            "n_nan": n_nan,
            "n_nonpos": n_nonpos,
            "n_kept": n_kept,
            "slices": slices,
        }
        return np.vstack(points_list), np.concatenate(values_list), report

    def _axes(self):
        return (
            self._log_mass_ax, self._log_teq_ax,
            self._log_R_ax, self._log_Luv_ax,
        )

    def _log_mdot(self, log_q):
        """log10 of the mass-loss rate in g/s.

        Inside the table this is the regular-grid interpolant. Outside it,
        the change from the nearest in-table point is

            energy-limited:  dlog Mdot = -dlog M + 4 dlog Teq + 3 dlog R + dlog L
            Parker wind:     Mdot proportional to R_s^2 * sqrt(T) * exp(-2 R_s / R)

        with R_s = GM / (2 c_s^2). Energy-limited applies for R < R_trans
        and the Parker wind for R >= R_trans.
        """
        log_q = np.atleast_2d(np.asarray(log_q, dtype=float))
        log_lin = np.asarray(self._rgi(log_q), dtype=float)

        axes = self._axes()
        overflow = np.zeros(len(log_q))
        log_ref = np.array(log_q, dtype=float, copy=True)
        for j, ax in enumerate(axes):
            overflow = np.maximum(
                overflow,
                np.maximum(ax[0] - log_q[:, j], 0.0)
                + np.maximum(log_q[:, j] - ax[-1], 0.0),
            )
            log_ref[:, j] = np.clip(log_q[:, j], ax[0], ax[-1])

        outside = overflow > 0.0
        if not np.any(outside):
            return log_lin.ravel()

        log_ref_val = np.asarray(self._rgi(log_ref), dtype=float)
        dlogM = log_q[:, 0] - log_ref[:, 0]
        dlogT = log_q[:, 1] - log_ref[:, 1]
        dlogR = log_q[:, 2] - log_ref[:, 2]
        dlogL = log_q[:, 3] - log_ref[:, 3]
        dlog_el = -dlogM + 4.0 * dlogT + 3.0 * dlogR + dlogL

        M = 10.0 ** log_q[:, 0]
        T = 10.0 ** log_q[:, 1]
        R = np.maximum(10.0 ** log_q[:, 2] / _R_EARTH_CM, 1e-6)
        M_ref = 10.0 ** log_ref[:, 0]
        T_ref = 10.0 ** log_ref[:, 1]
        R_ref = np.maximum(10.0 ** log_ref[:, 2] / _R_EARTH_CM, 1e-6)
        Rs = _R_sonic_earth(M, T)
        Rs_ref = _R_sonic_earth(M_ref, T_ref)
        dlog_parker = (
            2.0 * np.log10(Rs / Rs_ref)
            + 0.5 * np.log10(T / T_ref)
            + (-2.0 * Rs / R + 2.0 * Rs_ref / R_ref) / _LN10
        )

        Rtr = np.maximum(_R_trans_earth(M, T), 1e-12)
        x = np.log10(np.maximum(R, 1e-12) / Rtr)
        w = np.where(x >= 0.0, 1.0, 0.0)
        log_phys = log_ref_val + (1.0 - w) * dlog_el + w * dlog_parker
        return np.where(outside, log_phys, log_lin).ravel()

    def evaluate(self, planet_mass_earth, teq_k, transit_radius_cm, luv_erg_s):
        """Mass-loss rate in g/s for one planet.

        Inside the table this is the interpolant. Outside it, the rate is
        the energy-limited or Parker continuation, switching at R_trans.
        """
        query = np.array([[
            np.log10(float(planet_mass_earth)),
            np.log10(float(teq_k)),
            np.log10(float(transit_radius_cm)),
            np.log10(float(luv_erg_s)),
        ]])
        return float(10.0 ** self._log_mdot(query)[0])

    def evaluate_batch(self, planet_mass_earth, teq_k, transit_radius_cm, luv_erg_s):
        """Mass-loss rates in g/s for arrays of equal length."""
        queries = np.column_stack([
            np.log10(np.asarray(planet_mass_earth, dtype=float)),
            np.log10(np.asarray(teq_k,             dtype=float)),
            np.log10(np.asarray(transit_radius_cm,  dtype=float)),
            np.log10(np.asarray(luv_erg_s,          dtype=float)),
        ])
        return 10.0 ** self._log_mdot(queries)


# One shared grid, so repeated calls do not rebuild the tables.

_DEFAULT_GRID = None


def get_default_aiolos_grid(directory=None):
    """Return the grid built from the default table directory.

    A later call with the same directory reuses the grid already in memory.
    """
    global _DEFAULT_GRID
    if directory is None:
        directory = DEFAULT_DIRECTORY
    directory = _resolve_directory(directory)
    if _DEFAULT_GRID is None or _DEFAULT_GRID.directory != directory:
        _DEFAULT_GRID = AiolosMassLossGrid(directory=directory)
    return _DEFAULT_GRID
