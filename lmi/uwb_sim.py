from __future__ import annotations
from typing import Optional
import numpy as np
from .config import C0, DT, NT, T0, FP, TX_POS, RX_POS, PAIRS, N_PAIRS, Soil, Band, HIGH_BAND, F_REF_LOSS
from .scene import Scatterers, gamma
from .sar import refracted_path

def ricker_spectrum(f, fp: float=FP):
    f = np.asarray(f, float)
    return 2.0 / np.sqrt(np.pi) * f ** 2 / fp ** 3 * np.exp(-(f / fp) ** 2)

def simulate(scat: Scatterers, positions, soil: Soil, ground: bool=True, coupling: bool=True, noise_db: Optional[float]=-50.0, rng: Optional[np.random.Generator]=None, chunk: int=8, aperture: float=0.35, band: Optional[Band]=None) -> np.ndarray:
    return simulate_bands(scat, positions, soil, [band or HIGH_BAND], ground, coupling, noise_db, rng, chunk, aperture)[0]

def simulate_bands(scat: Scatterers, positions, soil: Soil, bands, ground: bool=True, coupling: bool=True, noise_db: Optional[float]=-50.0, rng: Optional[np.random.Generator]=None, chunk: int=8, aperture: float=0.35):
    positions = np.atleast_2d(np.asarray(positions, float))
    P = len(positions)
    f = np.fft.rfftfreq(NT, DT)
    sel = (f > 50000000.0) & (f < 6500000000.0)
    fs = f[sel]
    Rs = [ricker_spectrum(fs, b.fp) for b in bands]
    h = positions[:, 2]
    txi, rxi = (PAIRS[:, 0], PAIRS[:, 1])
    Dpair = np.linalg.norm(TX_POS[txi] - RX_POS[rxi], axis=1)
    g_ground = gamma(1.0, soil.eps)
    base = np.zeros((P, N_PAIRS, len(fs)), complex)
    if coupling:
        amp = 0.15 / Dpair
        c = amp[:, None] * (np.exp(-2j * np.pi * fs * (T0 + Dpair[:, None] / C0)) - 0.4 * np.exp(-2j * np.pi * fs * (T0 + 3.5e-10 + Dpair[:, None] / C0)))
        base += c[None]
    if ground:
        path = np.hypot(Dpair[None, :], 2 * h[:, None])
        cos = 2 * h[:, None] / path
        amp = g_ground * cos ** 2 / path
        base += amp[..., None] * np.exp(-2j * np.pi * fs * (T0 + path / C0)[..., None])
    specs = [base * R for R in Rs]
    del base
    K = len(scat.gamma)
    if K:
        n = soil.n
        T2 = 1.0 - g_ground ** 2
        ants = np.vstack([TX_POS, RX_POS])
        d = np.maximum(-scat.xyz[:, 2], 0.0001)
        patches = [R * (1j * fs * n / C0) for R in Rs]
        fs32 = fs.astype(np.float32)
        lossy = soil.alpha_exp != 0
        if lossy:
            alpha_f32 = (soil.alpha * (fs / F_REF_LOSS) ** soil.alpha_exp).astype(np.float32)
        for c0 in range(0, P, chunk):
            pc = positions[c0:c0 + chunk]
            dist = np.hypot(scat.xyz[None, :, 0] - pc[:, None, 0], scat.xyz[None, :, 1] - pc[:, None, 1])
            act = np.nonzero((dist < aperture).any(axis=0))[0]
            if not len(act):
                continue
            ax = pc[:, None, 0, None] + ants[None, :, 0, None]
            ay = pc[:, None, 1, None] + ants[None, :, 1, None]
            D = np.hypot(scat.xyz[act, 0][None, None] - ax, scat.xyz[act, 1][None, None] - ay)
            hh = pc[:, 2, None, None]
            ra, rs = refracted_path(D, hh, d[act][None, None], n)
            t1 = (ra + n * rs) / C0
            if lossy:
                one = hh / ra / (ra + rs) * (D < aperture)
            else:
                one = hh / ra * np.exp(-soil.alpha * rs) / (ra + rs) * (D < aperture)
            tau = T0 + t1[:, txi] + t1[:, 2 + rxi]
            a = scat.gamma[act] * scat.area[act] * T2 * one[:, txi] * one[:, 2 + rxi]
            if lossy:
                L = (rs[:, txi] + rs[:, 2 + rxi]).astype(np.float32)
                E = np.exp(-2j * np.pi * (tau.astype(np.float32)[..., None] * fs32) - L[..., None] * alpha_f32)
            else:
                E = np.exp(-2j * np.pi * (tau.astype(np.float32)[..., None] * fs32))
            S = np.einsum('cpk,cpkf->cpf', a.astype(np.complex64), E)
            del E
            for spec, patch in zip(specs, patches):
                spec[c0:c0 + chunk] += S * patch
    out = []
    for spec in specs:
        full = np.zeros((P, N_PAIRS, len(f)), complex)
        full[..., sel] = spec / DT
        x = np.fft.irfft(full, NT, axis=-1)
        del full
        if noise_db is not None:
            rng = rng or np.random.default_rng()
            x = x + rng.standard_normal(x.shape) * (10 ** (noise_db / 20) * np.abs(x).max())
        out.append(x)
    return out
