from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.signal import hilbert
from . import sar, uwb_proc, uwb_sim
from .config import C0, DT, NT, Soil, Band, HIGH_BAND

def footprint_shape(E, grid: sar.Grid, peak, rel_db: float=-6.0) -> dict:
    thr = E[peak] * 10 ** (rel_db / 20)
    lab, _ = label(E >= thr)
    m = lab == lab[peak]
    X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
    w = E[m]
    x, y = (X[m], Y[m])
    mx, my = (np.average(x, weights=w), np.average(y, weights=w))
    cov = np.cov(np.vstack([x - mx, y - my]), aweights=w, bias=True)
    l2, l1 = np.clip(np.linalg.eigvalsh(cov), 1e-08, None)
    dx = abs(grid.x[1] - grid.x[0])
    dy = abs(grid.y[1] - grid.y[0])
    l1c, l2c = (l1, l2)
    return dict(aspect=float(np.sqrt(l1c / l2c)), major_m=float(4 * np.sqrt(l1c)), minor_m=float(4 * np.sqrt(l2c)), area_m2=float(m.sum() * dx * dy), center=(float(mx), float(my)))

def column_profile(a, pos, soil: Soil, xy, z, offset: float=0.005) -> np.ndarray:
    g = sar.Grid(x=xy[0] + offset * np.array([-1.0, 0.0, 1.0]), y=xy[1] + offset * np.array([-1.0, 0.0, 1.0]), z=np.asarray(z, float))
    return sar.backproject(a, pos, soil, g).sum(axis=(0, 1))

def smooth_map(E, grid: sar.Grid, sigma_m: float) -> np.ndarray:
    return gaussian_filter(E, sigma_m / abs(grid.x[1] - grid.x[0]), mode='nearest')

def echo_layers(prof, z, rel_db: float=-12.0, top_rel_db: float=None) -> dict:
    env = np.abs(prof)
    thr = env.max() * 10 ** (rel_db / 20)
    pk = [i for i in range(1, len(env) - 1) if env[i] >= thr and env[i] >= env[i - 1] and (env[i] > env[i + 1])]
    if not pk:
        pk = [int(env.argmax())]
    if top_rel_db is not None:
        strong = [i for i in pk if env[i] >= env.max() * 10 ** (top_rel_db / 20)]
        pk = [i for i in pk if i >= strong[0]] if strong else pk
    return dict(top_z=float(z[pk[0]]), bottom_z=float(z[pk[-1]]), n_peaks=len(pk), top_amp=float(env[pk[0]]), bottom_amp=float(env[pk[-1]]))
_REF = {}

def _reference_waveform(band: Band=None):
    key = None if band is None else (band.fp, band.lo, band.hi)
    if key not in _REF:
        fp = HIGH_BAND.fp if band is None else band.fp
        f = np.fft.rfftfreq(NT, DT)
        x = np.fft.irfft(uwb_sim.ricker_spectrum(f, fp) * 1j * f * np.exp(-2j * np.pi * f * 3e-09) / DT, NT)
        _REF[key] = hilbert(uwb_proc.bandpass(x[None, None], band)[0, 0])
    return _REF[key]

def _centroid(p, dt):
    n = 4096
    P = np.abs(np.fft.fft(p, n)) ** 2
    f = np.fft.fftfreq(n, dt)
    sel = (f > 300000000.0) & (f < 6000000000.0)
    return float((f[sel] * P[sel]).sum() / P[sel].sum())

def spectral_centroid_ratio(prof, z, top_z, soil: Soil, half_window: float=None, band: Band=None) -> float:
    z = np.asarray(z, float)
    dz = z[1] - z[0]
    dt = 2 * dz * soil.n / C0
    if half_window is None:
        half_window = 4.5e-10 * soil.v / 2
    win = np.abs(z - top_z) <= half_window
    taper = np.hanning(win.sum())
    p = prof[win] * taper
    ref = _reference_waveform(band)
    t = 3e-09 + 2 * (z[win] - top_z) * soil.n / C0
    tt = np.arange(NT) * DT
    r = (np.interp(t, tt, ref.real) + 1j * np.interp(t, tt, ref.imag)) * taper
    return _centroid(p, dt) / _centroid(r, dt)
