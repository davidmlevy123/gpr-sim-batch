from __future__ import annotations
import numpy as np
from scipy.signal import bilinear, butter, lfilter, sosfilt
from scipy.signal.windows import tukey
from .config import CHIRP_DUR, CHIRP_F, FS_VIB, MMWAVE_LAMBDA, QUIET_DUR, Soil
V_BG = 3e-05
CHASSIS = 100.0
FOOTPRINT_SIGMA = 0.045
ADXL_NOISE = 2.25e-05 * 9.81 * np.sqrt(FS_VIB / 2)
LOW_BAND, HIGH_BAND = ((100.0, 600.0), (650.0, 800.0))

def log_chirp(fs=FS_VIB, f0=CHIRP_F[0], f1=CHIRP_F[1], dur=CHIRP_DUR, quiet=QUIET_DUR):
    n = int(round(dur * fs))
    t = np.arange(n) / fs
    k = np.log(f1 / f0)
    u = np.sin(2 * np.pi * f0 * dur / k * (np.exp(t * k / dur) - 1)) * tukey(n, 0.1)
    u /= np.abs(u).max()
    return np.concatenate([u, np.zeros(int(round(quiet * fs)))])

def _sdof(f0, q, gain, fs=FS_VIB):
    w0 = 2 * fs * np.tan(np.pi * f0 / fs)
    return bilinear([gain * w0 / q, 0.0], [1.0, w0 / q, w0 ** 2], fs)

def rls(x, d, taps: int, lam: float=1.0, delta: float=1e-06):
    x = np.asarray(x, float)
    d = np.asarray(d, float)
    sx = np.std(x) or 1.0
    xn = x / sx
    w = np.zeros(taps)
    Pm = np.eye(taps) / delta
    buf = np.zeros(taps)
    err = np.empty(len(x))
    for i in range(len(x)):
        buf[1:] = buf[:-1]
        buf[0] = xn[i]
        Px = Pm @ buf
        k = Px / (lam + buf @ Px)
        e = d[i] - w @ buf
        w += k * e
        Pm = (Pm - np.outer(k, Px)) / lam
        err[i] = e
    return (w / sx, err)

class VibroSim:

    def __init__(self, soil: Soil, rng: np.random.Generator, phase_noise: float=0.0001, hand_motion: float=0.0003, fs: float=FS_VIB, oversample: int=1):
        self.soil, self.rng, self.fs = (soil, rng, fs)
        self.oversample = oversample
        self.phase_noise, self.hand_motion = (phase_noise, hand_motion)
        self.u = log_chirp(fs)
        self._bg = _sdof(1800.0, 0.7, 1.0, fs)
        self._ch_radar = [_sdof(180.0, 5.0, 1.0, fs), _sdof(520.0, 8.0, 0.6, fs)]
        self._ch_adxl = [_sdof(180.0, 5.0, 1.1, fs), _sdof(520.0, 8.0, 0.5, fs)]
        self._hp = butter(4, 60.0, btype='highpass', fs=fs, output='sos')

    def soil_velocity(self, targets, xy) -> np.ndarray:
        u = self.u
        v_bg = lfilter(*self._bg, u)
        v = v_bg.copy()
        s2f = FOOTPRINT_SIGMA ** 2
        for t in targets:
            R = t.footprint_radius
            dist2 = (t.x - xy[0]) ** 2 + (t.y - xy[1]) ** 2
            w = R ** 2 / (R ** 2 + 2 * s2f) * np.exp(-dist2 / (2 * (s2f + R ** 2 / 2)))
            if w < 0.0001:
                continue
            if t.f0:
                coupling = np.exp(-t.depth / 0.12)
                resp = lfilter(*_sdof(t.f0, t.q / self.soil.acoustic_damping, t.gain * coupling, self.fs), u)
            else:
                resp = t.compliance * v_bg
            v += w * (resp - v_bg)
        return V_BG * v

    def _frame(self, modes, u):
        return CHASSIS * V_BG * sum((lfilter(b, a, u) for b, a in modes))

    def measure(self, v_soil, pre_roll: float=1.0):
        N = len(self.u)
        n0 = int(pre_roll * self.fs)
        u = np.concatenate([np.zeros(n0), self.u])
        vs = np.concatenate([np.zeros(n0), v_soil])
        t = np.arange(n0 + N) / self.fs
        hand = self.hand_motion * np.sin(2 * np.pi * 1.2 * t + self.rng.uniform(0, 2 * np.pi))
        disp = np.cumsum(vs - self._frame(self._ch_radar, u)) / self.fs + hand
        phi = 4 * np.pi * disp / MMWAVE_LAMBDA + self.rng.normal(0, self.phase_noise / np.sqrt(self.oversample), n0 + N)
        v_raw = np.gradient(np.unwrap(phi)) * self.fs * MMWAVE_LAMBDA / (4 * np.pi)
        acc = np.gradient(self._frame(self._ch_adxl, u)) * self.fs + self.rng.normal(0, ADXL_NOISE, n0 + N)
        return (sosfilt(self._hp, v_raw)[n0:], sosfilt(self._hp, acc)[n0:])

def calibrate(sim: VibroSim, n_chirps: int=5, taps: int=64) -> np.ndarray:
    zero = np.zeros_like(sim.u)
    raws, accs = zip(*(sim.measure(zero) for _ in range(n_chirps)))
    w, _ = rls(np.concatenate(accs), np.concatenate(raws), taps)
    return w

def resonance_score(v, u, fs=FS_VIB) -> float:
    f = np.fft.rfftfreq(len(u), 1 / fs)
    H2 = np.abs(np.fft.rfft(v)) ** 2 / (np.abs(np.fft.rfft(u)) ** 2 + 1e-12)
    lo = (f >= LOW_BAND[0]) & (f <= LOW_BAND[1])
    hi = (f >= HIGH_BAND[0]) & (f <= HIGH_BAND[1])
    return float(H2[lo].mean() / H2[hi].mean())

def score_at(sim: VibroSim, w, targets, xy) -> float:
    raw, acc = sim.measure(sim.soil_velocity(targets, xy))
    return resonance_score(raw - lfilter(w, 1.0, acc), sim.u, sim.fs)

def grid_scores(xs, ys, vals, gx, gy, sigma=0.03):
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    W = np.exp(-((GX[..., None] - xs) ** 2 + (GY[..., None] - ys) ** 2) / (2 * sigma ** 2))
    ws = W.sum(-1)
    out = (W * vals).sum(-1) / np.maximum(ws, 1e-12)
    out[ws < 0.001] = np.median(vals)
    return out

def dwell_velocity(sim: VibroSim, w, targets, xy, n_chirps: int=10) -> np.ndarray:
    v_soil = sim.soil_velocity(targets, xy)
    acc = np.zeros_like(sim.u)
    for _ in range(n_chirps):
        raw, a = sim.measure(v_soil)
        acc += raw - lfilter(w, 1.0, a)
    return acc / n_chirps

def dwell_score(sim: VibroSim, w, targets, xy, n_chirps: int=10) -> float:
    return resonance_score(dwell_velocity(sim, w, targets, xy, n_chirps), sim.u, sim.fs)
