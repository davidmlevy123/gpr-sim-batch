from __future__ import annotations
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from .config import BAND, C0, DT, NT, T0, TX_POS, RX_POS, PAIRS, Soil, Band
_SOS = butter(4, BAND, btype='bandpass', fs=1 / DT, output='sos')
_DPAIR = np.linalg.norm(TX_POS[PAIRS[:, 0]] - RX_POS[PAIRS[:, 1]], axis=1)
H_REF = 0.1

def ground_delay(h) -> np.ndarray:
    h = np.atleast_1d(np.asarray(h, float))
    return T0 + np.hypot(_DPAIR[None, :], 2 * h[:, None]) / C0

def remove_background(x):
    x = x - x.mean(axis=-1, keepdims=True)
    return x - x.mean(axis=0, keepdims=True)
_SOS_CACHE = {}

def bandpass(x, band: Band=None):
    if band is None:
        sos = _SOS
    else:
        key = (band.lo, band.hi)
        if key not in _SOS_CACHE:
            _SOS_CACHE[key] = butter(4, key, btype='bandpass', fs=1 / DT, output='sos')
        sos = _SOS_CACHE[key]
    return sosfiltfilt(sos, x, axis=-1)

def align_ground(a, h, inverse: bool=False):
    shift = ground_delay(h) - ground_delay(H_REF)
    f = np.fft.fftfreq(NT, DT)
    sgn = -1.0 if inverse else 1.0
    X = np.fft.fft(a, axis=-1)
    for c in range(0, X.shape[0], 256):
        X[c:c + 256] *= np.exp(sgn * 2j * np.pi * f * shift[c:c + 256, :, None])
    out = np.fft.ifft(X, axis=-1)
    del X
    return out if np.iscomplexobj(a) else out.real

def svd_suppress(a, window: int=40, modes: int=2, min_spread: float=0.6, copy: bool=True, gate_ns: float=None):
    if copy:
        a = np.array(a, copy=True)
    P = a.shape[0]
    g_idx = np.round(ground_delay(H_REF)[0] / DT).astype(int)
    half = None if gate_ns is None else int(round(gate_ns * 1e-09 / DT))
    edges = list(range(0, P, window)) + [P]
    if len(edges) > 2 and edges[-1] - edges[-2] < window // 2:
        edges.pop(-2)
    for k in range(a.shape[1]):
        for s, e in zip(edges[:-1], edges[1:]):
            cols = slice(None) if half is None else slice(max(g_idx[k] - half, 0), g_idx[k] + half + 1)
            B = a[s:e, k, cols]
            if e - s <= modes:
                a[s:e, k, cols] = 0
                continue
            U, S, Vh = np.linalg.svd(B, full_matrices=False)
            p2 = np.abs(U[:, :modes]) ** 2
            spread = 1.0 / (p2 ** 2).sum(axis=0) / (e - s)
            sel = np.nonzero(spread >= min_spread)[0]
            if len(sel):
                a[s:e, k, cols] = B - U[:, sel] * S[sel] @ Vh[sel]
    return a

def tvg_gain(h, soil: Soil, cap_db: float=60.0) -> np.ndarray:
    tg = ground_delay(h)[..., None]
    t = np.arange(NT) * DT
    g = np.exp(soil.alpha * soil.v * np.clip(t - tg, 0, None))
    return np.minimum(g, 10 ** (cap_db / 20))

def preprocess(traces, pos_est, soil_est: Soil, svd_window: int=40, svd_modes: int=2, tvg: bool=True, band: Band=None, svd_gate_ns: float=None) -> np.ndarray:
    h = np.asarray(pos_est)[:, 2]
    x = bandpass(remove_background(traces), band)
    a = hilbert(x, axis=-1)
    del x
    a = align_ground(a, h)
    a = svd_suppress(a, svd_window, svd_modes, copy=False, gate_ns=svd_gate_ns)
    a = align_ground(a, h, inverse=True)
    if tvg:
        for c in range(0, a.shape[0], 256):
            a[c:c + 256] *= tvg_gain(h[c:c + 256], soil_est)
    return a
