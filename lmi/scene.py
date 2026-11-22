from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.ndimage import uniform_filter1d
from .config import Soil

def gamma(eps1: float, eps2: float) -> float:
    n1, n2 = (np.sqrt(eps1), np.sqrt(eps2))
    return float((n1 - n2) / (n1 + n2))

@dataclass
class Scatterers:
    xyz: np.ndarray
    gamma: np.ndarray
    area: np.ndarray

    @staticmethod
    def empty() -> 'Scatterers':
        return Scatterers(np.zeros((0, 3)), np.zeros(0), np.zeros(0))

    @staticmethod
    def concat(parts) -> 'Scatterers':
        parts = [p for p in parts if len(p.gamma)]
        if not parts:
            return Scatterers.empty()
        return Scatterers(np.vstack([p.xyz for p in parts]), np.concatenate([p.gamma for p in parts]), np.concatenate([p.area for p in parts]))

def _ellipse_grid(a: float, b: float, step: float):
    nx, ny = (int(np.ceil(a / step)), int(np.ceil(b / step)))
    gx = (np.arange(-nx, nx) + 0.5) * step
    gy = (np.arange(-ny, ny) + 0.5) * step
    X, Y = np.meshgrid(gx, gy, indexing='ij')
    r2 = (X / a) ** 2 + (Y / b) ** 2
    m = r2 <= 1.0
    if not m.any():
        return (np.array([0.0]), np.array([0.0]), np.pi * a * b, np.array([1.0]))
    return (X[m], Y[m], step * step, np.sqrt(1.0 - r2[m]))

@dataclass
class DiscMine:
    name: str
    x: float
    y: float
    depth: float
    radius: float
    height: float
    eps_body: float = 2.95
    cav_radius: float = 0.0
    cav_top: float = 0.0
    cav_thick: float = 0.0
    f0: Optional[float] = 350.0
    q: float = 6.0
    gain: float = 8.0
    compliance: float = 1.0
    kind: str = 'mine'

    @property
    def footprint_radius(self) -> float:
        return self.radius

    def scatterers(self, soil: Soil, step: float=0.006) -> Scatterers:
        ns, nb = (soil.n, np.sqrt(self.eps_body))
        parts = []

        def layer(radius, extra_depth, g):
            dx, dy, dA, _ = _ellipse_grid(radius, radius, step)
            k = len(dx)
            parts.append(Scatterers(np.column_stack([self.x + dx, self.y + dy, np.full(k, -(self.depth + extra_depth))]), np.full(k, g), np.full(k, dA)))
        g_top = gamma(soil.eps, self.eps_body)
        layer(self.radius, 0.0, g_top)
        T = 1.0 - g_top ** 2
        if self.cav_thick > 0:
            za = self.cav_top * nb / ns
            g_a = gamma(self.eps_body, 1.0)
            layer(self.cav_radius, za, g_a * T)
            g_b = gamma(1.0, self.eps_body)
            layer(self.cav_radius, za + self.cav_thick / ns, g_b * T * (1 - g_a ** 2))
        layer(self.radius, self.height * nb / ns, gamma(self.eps_body, soil.eps) * T)
        return Scatterers.concat(parts)

@dataclass
class Ellipsoid:
    name: str
    x: float
    y: float
    depth: float
    a: float
    b: float
    c: float
    eps: float
    f0: Optional[float] = None
    q: float = 1.0
    gain: float = 0.0
    compliance: float = 0.6
    kind: str = 'clutter'
    yaw: float = 0.0

    @property
    def footprint_radius(self) -> float:
        return float(np.sqrt(self.a * self.b))

    def scatterers(self, soil: Soil, step: float=0.006) -> Scatterers:
        ns, nr = (soil.n, np.sqrt(self.eps))
        dx, dy, dA, s = _ellipse_grid(self.a, self.b, step)
        k = len(dx)
        z_top = self.depth + self.c * (1 - s)
        z_bot = z_top + 2 * self.c * s * nr / ns
        g1 = gamma(soil.eps, self.eps)
        g2 = gamma(self.eps, soil.eps) * (1 - g1 ** 2)
        c, s_ = (np.cos(self.yaw), np.sin(self.yaw))
        xy = np.column_stack([self.x + c * dx - s_ * dy, self.y + s_ * dx + c * dy])
        return Scatterers(np.vstack([np.column_stack([xy, -z_top]), np.column_stack([xy, -z_bot])]), np.concatenate([np.full(k, g1), np.full(k, g2)]), np.full(2 * k, dA))

def vs50(x, y, depth) -> DiscMine:
    return DiscMine('VS-50', x, y, depth, radius=0.045, height=0.045, cav_radius=0.035, cav_top=0.008, cav_thick=0.006, f0=350.0, q=6.0, gain=8.0)

def pmn2(x, y, depth) -> DiscMine:
    return DiscMine('PMN-2', x, y, depth, radius=0.0625, height=0.054, cav_radius=0.05, cav_top=0.01, cav_thick=0.008, f0=280.0, q=6.0, gain=8.0)

def tm62p3(x, y, depth) -> DiscMine:
    return DiscMine('TM-62P3', x, y, depth, radius=0.16, height=0.128, cav_radius=0.12, cav_top=0.012, cav_thick=0.005, f0=150.0, q=5.0, gain=10.0)

def granite_rock(x, y, depth) -> Ellipsoid:
    return Ellipsoid('granite rock', x, y, depth, a=0.045, b=0.045, c=0.0225, eps=5.5)

class Scene:

    def __init__(self, targets, n_pebbles: int=0, extent=((-0.3, 0.3), (-0.3, 0.3)), rng: Optional[np.random.Generator]=None, keep_out: float=0.09):
        self.targets = list(targets)
        rng = rng or np.random.default_rng()
        self.pebbles = []
        while len(self.pebbles) < n_pebbles:
            x, y = (rng.uniform(*extent[0]), rng.uniform(*extent[1]))
            if any((np.hypot(x - t.x, y - t.y) < keep_out for t in self.targets)):
                continue
            r = rng.uniform(0.005, 0.014)
            self.pebbles.append(Ellipsoid('pebble', x, y, rng.uniform(0.005, 0.12), r, r * rng.uniform(0.6, 1.0), r * 0.6, rng.uniform(4.5, 7.5)))

    def scatterers(self, soil: Soil, step: float=0.006) -> Scatterers:
        return Scatterers.concat([o.scatterers(soil, step) for o in self.targets + self.pebbles])

@dataclass
class Sweep:
    t: np.ndarray
    pos_true: np.ndarray
    pos_est: np.ndarray

def serpentine(x_range, y_range, line_spacing, speed, frame_rate, h0, rng, vio_sigma: float=0.001, turn_time: float=0.3) -> Sweep:
    xs = np.arange(x_range[0], x_range[1] + 1e-09, line_spacing)
    step = speed / frame_rate
    L = y_range[1] - y_range[0]
    n = int(round(L / speed * frame_rate))
    ts, pts = ([], [])
    t_now = 0.0
    for i, x in enumerate(xs):
        y = y_range[0] + np.arange(n) * step
        if i % 2:
            y = y[::-1]
        ts.append(t_now + np.arange(n) / frame_rate)
        pts.append(np.column_stack([np.full(n, x), y]))
        t_now = ts[-1][-1] + turn_time
    t = np.concatenate(ts)
    xy = np.vstack(pts)
    ph = rng.uniform(0, 2 * np.pi, 2)
    h = h0 + 0.012 * np.sin(2 * np.pi * t / 1.9 + ph[0]) + 0.004 * np.sin(2 * np.pi * t / 0.37 + ph[1])
    pos_true = np.column_stack([xy, h])
    err = uniform_filter1d(rng.standard_normal(pos_true.shape), 9, axis=0) * 3.0 * vio_sigma
    return Sweep(t, pos_true, pos_true + err)
