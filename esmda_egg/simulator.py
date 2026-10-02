"""
Minimal 2D incompressible oil-water waterflood simulator (IMPES).

IMPES = IMplicit Pressure, Explicit Saturation:
  1. Solve an implicit (linear) pressure equation for the current saturation.
  2. Move water explicitly with upwind fluxes, using small CFL-limited substeps.

Governing equations (incompressible fluids and rock, no gravity or capillarity):
  pressure:    -div( lambda_t * K * grad p ) = q_t
  saturation:  phi * dSw/dt + div( f_w * v ) = q_w

  K        = absolute permeability [m2]
  lambda_t = total mobility = krw/mu_w + kro/mu_o [1/(Pa.s)]
  f_w      = water fractional flow = (krw/mu_w) / lambda_t [-]
  v        = Darcy (total) velocity [m/s]

Wells: injectors at a fixed water rate, producers at a fixed bottom-hole pressure
(BHP), both coupled to their grid cell with a Peaceman well index (WI).

Units are SI internally; inputs/outputs use field-friendly units (mD, m3/day, bar).
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

MD = 9.869233e-16      # 1 milliDarcy in m2
DAY = 86400.0          # seconds
BAR = 1e5              # Pa

# Egg Model well locations, (i, j) 1-based grid indices on the 60 x 60 grid
# (Jansen et al., 2014, Geoscience Data Journal).
EGG_INJECTORS = {
    "INJECT1": (5, 57), "INJECT2": (30, 53), "INJECT3": (2, 35), "INJECT4": (27, 29),
    "INJECT5": (50, 35), "INJECT6": (8, 9), "INJECT7": (32, 2), "INJECT8": (57, 6),
}
EGG_PRODUCERS = {
    "PROD1": (16, 43), "PROD2": (35, 40), "PROD3": (23, 16), "PROD4": (43, 18),
}


@dataclass
class ReservoirConfig:
    nx: int = 60
    ny: int = 60
    dx: float = 8.0               # m (Egg Model cell size)
    dy: float = 8.0               # m
    h: float = 28.0               # m (7 Egg layers x 4 m, collapsed to 2D)
    porosity: float = 0.2
    mu_w: float = 1.0e-3          # Pa.s (1 cP)
    mu_o: float = 5.0e-3          # Pa.s (5 cP)
    swc: float = 0.2              # connate water saturation
    sor: float = 0.1              # residual oil saturation
    krw0: float = 0.75            # end-point water relative permeability
    kro0: float = 0.8             # end-point oil relative permeability
    nw: float = 3.0               # Corey exponent, water
    no: float = 3.0               # Corey exponent, oil
    inj_rate: float = 79.5        # m3/day of water per injector
    prod_bhp: float = 395.0       # bar
    rw: float = 0.1               # m, wellbore radius
    active: np.ndarray = None     # optional boolean mask (ny, nx); False = inactive cell
    injectors: dict = field(default_factory=lambda: dict(EGG_INJECTORS))
    producers: dict = field(default_factory=lambda: dict(EGG_PRODUCERS))


class WaterfloodSimulator:
    def __init__(self, cfg: ReservoirConfig):
        self.cfg = cfg
        c = cfg
        self.n = c.nx * c.ny
        idx = np.arange(self.n).reshape(c.ny, c.nx)
        # Cell-to-cell connections in x and y (row-major cell numbering: cell = j*nx + i)
        self.cx = (idx[:, :-1].ravel(), idx[:, 1:].ravel())
        self.cy = (idx[:-1, :].ravel(), idx[1:, :].ravel())
        self.geo_x = c.h * c.dy / c.dx   # area / distance for x-connections
        self.geo_y = c.h * c.dx / c.dy
        act = np.ones((c.ny, c.nx), bool) if c.active is None else c.active.astype(bool)
        self.pv = c.porosity * c.dx * c.dy * c.h * np.where(act.ravel(), 1.0, 1e-3)  # pore volume [m3]

        def cell(ij):
            i, j = ij
            return (j - 1) * c.nx + (i - 1)
        self.inj_names = list(c.injectors)
        self.prod_names = list(c.producers)
        self.inj_cells = np.array([cell(c.injectors[w]) for w in self.inj_names])
        self.prod_cells = np.array([cell(c.producers[w]) for w in self.prod_names])
        # Peaceman equivalent radius for a square, isotropic cell
        self.r0 = 0.14 * np.sqrt(c.dx ** 2 + c.dy ** 2)

    # ---- rock-fluid functions -------------------------------------------------
    def _mobilities(self, sw):
        c = self.cfg
        se = np.clip((sw - c.swc) / (1.0 - c.swc - c.sor), 0.0, 1.0)  # normalised saturation
        lw = c.krw0 * se ** c.nw / c.mu_w
        lo = c.kro0 * (1.0 - se) ** c.no / c.mu_o
        return lw, lo

    def _max_dfw(self):
        """Largest slope of the fractional-flow curve (sets the CFL limit)."""
        s = np.linspace(self.cfg.swc, 1 - self.cfg.sor, 400)
        lw, lo = self._mobilities(s)
        fw = lw / (lw + lo)
        return np.max(np.abs(np.gradient(fw, s)))

    # ---- main run --------------------------------------------------------------
    def run(self, perm_md, report_days, pressure_dt_days=30.0):
        """
        Simulate the waterflood.

        perm_md      : array (ny, nx) or (n,) of permeability in mD
        report_days  : increasing array of report times [days]
        returns dict with per-well arrays, shape (n_reports, n_wells):
            'qo', 'qw'  producer oil and water rates [m3/day]
            'bhp_inj'   injector bottom-hole pressure [bar]
            'sw'        final water saturation (ny, nx)
        """
        c = self.cfg
        k = np.asarray(perm_md, float).ravel() * MD
        n = self.n
        sw = np.full(n, c.swc)
        # Harmonic-mean permeability on each face, times geometry factor
        tx = self.geo_x * 2.0 / (1.0 / k[self.cx[0]] + 1.0 / k[self.cx[1]])
        ty = self.geo_y * 2.0 / (1.0 / k[self.cy[0]] + 1.0 / k[self.cy[1]])
        wi_geo = 2 * np.pi * c.h / np.log(self.r0 / c.rw)
        wi_inj = wi_geo * k[self.inj_cells]
        wi_prod = wi_geo * k[self.prod_cells]
        q_inj = c.inj_rate / DAY
        p_bhp = c.prod_bhp * BAR
        dfw_max = self._max_dfw()

        a_idx = np.concatenate([self.cx[0], self.cy[0]])
        b_idx = np.concatenate([self.cx[1], self.cy[1]])
        t_face = np.concatenate([tx, ty])

        nr = len(report_days)
        out = {"qo": np.zeros((nr, len(self.prod_cells))),
               "qw": np.zeros((nr, len(self.prod_cells))),
               "bhp_inj": np.zeros((nr, len(self.inj_cells)))}
        t = 0.0
        for r, t_rep in enumerate(report_days):
            while t < t_rep - 1e-9:
                dt_p = min(pressure_dt_days, t_rep - t)
                # --- 1. implicit pressure solve -------------------------------
                lw, lo = self._mobilities(sw)
                lt = lw + lo
                lt_face = 0.5 * (lt[a_idx] + lt[b_idx])
                T = t_face * lt_face
                diag = np.bincount(a_idx, T, n) + np.bincount(b_idx, T, n)
                rhs = np.zeros(n)
                jp = wi_prod * lt[self.prod_cells]           # producer productivity
                np.add.at(diag, self.prod_cells, jp)
                np.add.at(rhs, self.prod_cells, jp * p_bhp)
                np.add.at(rhs, self.inj_cells, q_inj)
                A = sp.csr_matrix(
                    (np.concatenate([diag, -T, -T]),
                     (np.concatenate([np.arange(n), a_idx, b_idx]),
                      np.concatenate([np.arange(n), b_idx, a_idx]))), shape=(n, n))
                p = spla.spsolve(A.tocsc(), rhs, permc_spec="MMD_AT_PLUS_A")
                flux = T * (p[a_idx] - p[b_idx])             # +ve: a -> b  [m3/s]
                q_prod = jp * (p[self.prod_cells] - p_bhp)   # +ve: produced [m3/s]
                # --- 2. explicit upwind saturation transport -----------------
                outflow = (np.bincount(a_idx, np.maximum(flux, 0), n)
                           + np.bincount(b_idx, np.maximum(-flux, 0), n))
                np.add.at(outflow, self.prod_cells, q_prod)
                dt_cfl = 0.9 * np.min(self.pv / (outflow * dfw_max + 1e-30))
                tau, dt_tot = 0.0, dt_p * DAY
                up = np.where(flux >= 0, a_idx, b_idx)       # upwind cell for each face
                while tau < dt_tot - 1e-6:
                    dts = min(dt_cfl, dt_tot - tau)
                    lw, lo = self._mobilities(sw)
                    fw = lw / (lw + lo)
                    fwf = fw[up] * flux
                    dw = np.bincount(b_idx, fwf, n) - np.bincount(a_idx, fwf, n)
                    np.add.at(dw, self.inj_cells, q_inj)
                    np.add.at(dw, self.prod_cells, -fw[self.prod_cells] * q_prod)
                    sw = np.clip(sw + dts * dw / self.pv, c.swc, 1 - c.sor)
                    tau += dts
                t += dt_p
            # --- report well data at end of step ------------------------------
            lw, lo = self._mobilities(sw)
            fw = lw / (lw + lo)
            fwp = fw[self.prod_cells]
            out["qw"][r] = fwp * q_prod * DAY
            out["qo"][r] = (1 - fwp) * q_prod * DAY
            lt = lw + lo
            out["bhp_inj"][r] = (p[self.inj_cells] + q_inj / (wi_inj * lt[self.inj_cells])) / BAR
        out["sw"] = sw.reshape(c.ny, c.nx)
        return out
