from __future__ import annotations
import numpy as np
from .sar import Grid, polarity

def energy_map(vol) -> np.ndarray:
    return np.abs(vol).max(axis=-1)

def polarity_map(vol) -> np.ndarray:
    iz = np.abs(vol).argmax(axis=-1)
    return polarity(np.take_along_axis(vol, iz[..., None], -1)[..., 0])

def depth_map(vol, grid: Grid) -> np.ndarray:
    return grid.z[np.abs(vol).argmax(axis=-1)]

def _dist(grid: Grid, xy):
    X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
    return np.hypot(X - xy[0], Y - xy[1])

def peak_index(E, grid: Grid, xy, r_obj=0.06):
    m = np.where(_dist(grid, xy) <= r_obj, E, -np.inf)
    return np.unravel_index(np.argmax(m), E.shape)

def region_snr_db(E, grid: Grid, xy, exclude, r_obj=0.06, r_excl=0.1, edge=0.0) -> float:
    peak = E[peak_index(E, grid, xy, r_obj)]
    bg = np.ones(E.shape, bool)
    for o in exclude:
        bg &= _dist(grid, o) > r_excl
    if edge:
        X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
        bg &= (X > grid.x[0] + edge) & (X < grid.x[-1] - edge) & (Y > grid.y[0] + edge) & (Y < grid.y[-1] - edge)
    return float(20 * np.log10(peak / np.sqrt(np.mean(E[bg] ** 2))))

def mine_likeness(P, R) -> np.ndarray:
    return np.clip((1 + P) / 2, 0.01, 1.0) * R

def contrast_db(L, grid: Grid, xy_a, xy_b) -> float:
    ia = np.unravel_index(np.argmin(_dist(grid, xy_a)), L.shape)
    ib = np.unravel_index(np.argmin(_dist(grid, xy_b)), L.shape)
    return float(10 * np.log10(L[ia] / L[ib]))
