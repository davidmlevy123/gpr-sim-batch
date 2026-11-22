from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import hilbert
from .config import C0, DT, NT, T0, TX_POS, RX_POS, PAIRS, Soil

def refracted_path(D, h, d, n, iters: int=40):
    D, h, d = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (D, h, d)))
    d = np.maximum(d, 1e-07)
    if n == 1.0:
        s = D * h / (h + d)
    else:
        lo, hi = (np.zeros_like(D), D.copy())
        s = D * h / (h + d / n)
        for _ in range(iters):
            ra = np.hypot(s, h)
            rs = np.hypot(D - s, d)
            f = s / ra - n * (D - s) / rs
            neg = f < 0
            lo = np.where(neg, s, lo)
            hi = np.where(neg, hi, s)
            fp = h ** 2 / ra ** 3 + n * d ** 2 / rs ** 3
            s_new = s - f / fp
            bad = (s_new <= lo) | (s_new >= hi) | ~np.isfinite(s_new)
            s_next = np.where(bad, 0.5 * (lo + hi), s_new)
            done = np.max(np.abs(s_next - s), initial=0.0) < 1e-12
            s = s_next
            if done:
                break
    return (np.hypot(s, h), np.hypot(D - s, d))

class TravelTimeTable:

    def __init__(self, n, h_grid, d_grid, D_max, dD=0.001):
        self.n = n
        self.h = np.asarray(h_grid, float)
        self.dh = self.h[1] - self.h[0] if len(self.h) > 1 else 1.0
        self.dD = dD
        self.D = np.arange(0.0, D_max + 2 * dD, dD)
        ra, rs = refracted_path(self.D[None, None, :], self.h[:, None, None], np.asarray(d_grid, float)[None, :, None], n)
        self.T = (ra + n * rs) / C0

    def one_way(self, h, D):
        D = np.asarray(D, float)
        hi = (h - self.h[0]) / self.dh
        i0 = int(np.clip(np.floor(hi), 0, max(len(self.h) - 2, 0)))
        wh = float(np.clip(hi - i0, 0.0, 1.0)) if len(self.h) > 1 else 0.0
        Th = self.T[i0] * (1 - wh) + (self.T[i0 + 1] * wh if wh else 0.0)
        di = D / self.dD
        j0 = np.clip(np.floor(di).astype(int), 0, len(self.D) - 2)
        wd = np.clip(di - j0, 0.0, 1.0)
        return Th[:, j0] * (1 - wd) + Th[:, j0 + 1] * wd

@dataclass
class Grid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

def backproject(a, pos, soil: Soil, grid: Grid, aperture: float=0.15) -> np.ndarray:
    P = len(pos)
    hgrid = np.arange(pos[:, 2].min() - 0.002, pos[:, 2].max() + 0.003, 0.001)
    tab = TravelTimeTable(soil.n, hgrid, grid.z, aperture + 0.01)
    nx, ny, nz = (len(grid.x), len(grid.y), len(grid.z))
    vol = np.zeros((nx, ny, nz), complex)
    cnt = np.zeros((nx, ny))
    ants = np.vstack([TX_POS, RX_POS])
    ti, ri = (PAIRS[:, 0], len(TX_POS) + PAIRS[:, 1])
    reach = aperture + np.abs(ants).max()
    for p in range(P):
        hx, hy, hh = pos[p]
        x0, x1 = np.searchsorted(grid.x, [hx - reach, hx + reach])
        y0, y1 = np.searchsorted(grid.y, [hy - reach, hy + reach])
        if x1 <= x0 or y1 <= y0:
            continue
        X, Y = np.meshgrid(grid.x[x0:x1], grid.y[y0:y1], indexing='ij')
        D = np.hypot(X[None] - (hx + ants[:, 0, None, None]), Y[None] - (hy + ants[:, 1, None, None]))
        ok = D < aperture
        t = tab.one_way(hh, np.minimum(D, aperture))
        tau = T0 + t[:, ti] + t[:, ri]
        mask = ok[ti] & ok[ri]
        s = tau / DT
        i0 = np.floor(s).astype(int)
        valid = (i0 >= 0) & (i0 < NT - 1) & mask[None]
        i0 = np.clip(i0, 0, NT - 2)
        w = s - i0
        tr = a[p]
        k = np.arange(len(PAIRS))[None, :, None, None]
        val = tr[k, i0] * (1 - w) + tr[k, i0 + 1] * w
        val = np.where(valid, val, 0).sum(axis=1)
        vol[x0:x1, y0:y1] += np.moveaxis(val, 0, -1)
        cnt[x0:x1, y0:y1] += mask.sum(axis=0)
    return vol / np.maximum(cnt, 1)[..., None]
_REF_PHASE = None

def reference_phase() -> float:
    global _REF_PHASE
    if _REF_PHASE is None:
        from . import uwb_proc, uwb_sim
        f = np.fft.rfftfreq(NT, DT)
        tau = 3e-09
        x = np.fft.irfft(uwb_sim.ricker_spectrum(f) * 1j * f * np.exp(-2j * np.pi * f * tau) / DT, NT)
        z = hilbert(uwb_proc.bandpass(x[None, None])[0, 0])
        _REF_PHASE = float(np.angle(z[int(round(tau / DT))]))
    return _REF_PHASE

def polarity(vol: np.ndarray) -> np.ndarray:
    return np.cos(np.angle(vol) - reference_phase())
