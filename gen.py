import argparse
import dataclasses
import json
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import maximum_filter

from lmi import features, fusion, sar, scene, uwb_proc, uwb_sim, vibro
from lmi.config import DRY_SAND, HIGH_BAND, MEDIUM_LOAM, WET_CLAY

SOILS = [("sand", DRY_SAND), ("clay", WET_CLAY), ("loam", MEDIUM_LOAM)]
CELLS = []
for s in range(3):
    for k in ("vs50", "pmn2"):
        for d in ((0.02, 0.07), (0.07, 0.15)):
            CELLS += [(s, k, d)] * 3
    for d in ((0.02, 0.10), (0.10, 0.20), (0.20, 0.30)):
        CELLS += [(s, "tm62p3", d)] * 3
    for k in ("rock", "water_bottle", "empty_bottle", "can", "root"):
        for d in ((0.02, 0.10), (0.10, 0.20), (0.20, 0.30)):
            CELLS += [(s, k, d)] * 2
METAL_CELLS = []
for s in range(3):
    for d in ((0.02, 0.10), (0.10, 0.20), (0.20, 0.30)):
        METAL_CELLS += [(s, "tm62m", d)] * 3
    for d in ((0.02, 0.07), (0.07, 0.15)):
        METAL_CELLS += [(s, "ozm", d)] * 3
TIN_CELLS = []
for s in range(3):
    for d in ((0.02, 0.10), (0.10, 0.20), (0.20, 0.30)):
        TIN_CELLS += [(s, "steel_tin", d)] * 2
FRAG_CELLS = []
for s in range(3):
    for d in ((0.01, 0.05), (0.05, 0.10), (0.10, 0.20)):
        FRAG_CELLS += [(s, "fragment", d)] * 4
DECOY_CELLS = []
for s in range(3):
    for k in ("vs50_frag", "pmn2_frag"):
        for d in ((0.02, 0.07), (0.07, 0.15)):
            DECOY_CELLS += [(s, k, d)] * 2
SETS = {"train": (500000, 3), "test": (600000, 2), "test2": (700000, 2), "test3": (800000, 6),
        "metal_train": (900000, 4), "test4": (1000000, 4), "tin_train": (1100000, 8), "test5": (2000000, 70),
        "frag_train": (2100000, 4), "test6": (2200000, 10)}
LAYOUT = {"metal_train": [METAL_CELLS], "test4": [CELLS, METAL_CELLS], "tin_train": [TIN_CELLS],
          "test5": [CELLS + METAL_CELLS + TIN_CELLS], "frag_train": [FRAG_CELLS + DECOY_CELLS],
          "test6": [CELLS + METAL_CELLS + TIN_CELLS + FRAG_CELLS * 3 + DECOY_CELLS * 2]}


def cell(set_name, i):
    if set_name not in LAYOUT:
        return CELLS[i % len(CELLS)]
    mult = SETS[set_name][1]
    for cells in LAYOUT[set_name]:
        if i < len(cells) * mult:
            return cells[i % len(cells)]
        i -= len(cells) * mult
    raise IndexError


def set_size(set_name):
    return sum(len(c) for c in LAYOUT.get(set_name, [CELLS])) * SETS[set_name][1]
ZF = np.arange(0.01, 0.40, 0.001)
MINES = {"vs50", "pmn2", "tm62p3", "tm62m", "ozm", "vs50_frag", "pmn2_frag"}


def fragment(rng, x, y, depth):
    m = 10 ** rng.uniform(np.log10(0.5), np.log10(20.0))
    ba, ca = rng.uniform(0.3, 0.8), rng.uniform(0.15, 0.4)
    a = (m * 1e-3 / 7850.0 / (4.0 / 3.0 * np.pi * ba * ca)) ** (1.0 / 3.0)
    return scene.Ellipsoid("steel fragment", x, y, depth, a, a * ba, a * ca, 1e4, compliance=0.9,
                           yaw=rng.uniform(0, np.pi))


def make_object(kind, rng, depth):
    x, y = rng.uniform(-0.03, 0.03, 2)
    yaw = rng.uniform(0, np.pi)
    if kind == "vs50":
        return scene.vs50(x, y, depth)
    if kind == "pmn2":
        return scene.pmn2(x, y, depth)
    if kind == "tm62p3":
        return scene.tm62p3(x, y, depth)
    if kind == "tm62m":
        return scene.DiscMine("TM-62M", x, y, depth, radius=0.16, height=0.128, eps_body=1e4,
                              f0=200.0, q=5.0, gain=4.0)
    if kind == "ozm":
        return scene.DiscMine("OZM", x, y, depth, radius=0.054, height=0.17, eps_body=1e4,
                              f0=None, compliance=0.9)
    if kind == "fragment":
        return fragment(rng, x, y, depth)
    if kind == "rock":
        a = rng.uniform(0.03, 0.06)
        return scene.Ellipsoid("rock", x, y, depth, a, a * rng.uniform(0.7, 1.0), a * rng.uniform(0.4, 0.7),
                               rng.uniform(4.5, 7.5), yaw=yaw)
    if kind == "water_bottle":
        return scene.Ellipsoid("water bottle", x, y, depth, 0.10, 0.035, 0.035, 80.0, compliance=0.8, yaw=yaw)
    if kind == "empty_bottle":
        return scene.Ellipsoid("empty bottle", x, y, depth, 0.10, 0.035, 0.035, 1.05,
                               f0=rng.uniform(250, 600), q=4.0, gain=4.0, yaw=yaw)
    if kind == "steel_tin":
        return scene.Ellipsoid("crushed steel tin", x, y, depth, 0.05, 0.04, 0.025, 1e4, compliance=0.9, yaw=yaw)
    if kind == "can":
        return scene.Ellipsoid("crushed can", x, y, depth, 0.05, 0.035, 0.02, 1e4, compliance=0.9, yaw=yaw)
    return scene.Ellipsoid("wet root", x, y, depth, 0.10, 0.015, 0.015, 20.0, compliance=0.9, yaw=yaw)


def make_targets(kind, rng, depth):
    if not kind.endswith("_frag"):
        return [make_object(kind, rng, depth)]
    mine = make_object(kind[:-5], rng, depth)
    r, th = rng.uniform(0.05, 0.10), rng.uniform(0, 2 * np.pi)
    return [mine, fragment(rng, mine.x + r * np.cos(th), mine.y + r * np.sin(th), rng.uniform(0.01, 0.06))]


def dwell(soil, targets, xy, seed, n):
    sim = vibro.VibroSim(soil, np.random.default_rng(seed), phase_noise=1e-4, oversample=10)
    w = vibro.calibrate(sim, n_chirps=3)
    s = vibro.dwell_score(sim, w, targets, xy, n_chirps=n)
    ref = np.median([vibro.dwell_score(sim, w, [], (0.5, 0.5), n_chirps=n) for _ in range(3)])
    return float(s / ref)


def one(args):
    i, set_name = args
    base, _ = SETS[set_name]
    s, kind, (d0, d1) = cell(set_name, i)
    sname, soil0 = SOILS[s]
    soil = dataclasses.replace(soil0, alpha_exp=1.0)
    rng = np.random.default_rng(base + i)
    depth = float(rng.uniform(d0, d1))
    targets = make_targets(kind, rng, depth)
    obj = targets[0]
    R = max(getattr(obj, "footprint_radius", 0.05), getattr(obj, "a", 0.0))
    half = max(0.12, R + 0.08, 0.18 if len(targets) > 1 else 0.0)
    sc = scene.Scene(targets, n_pebbles=10, extent=((-half - 0.03, half + 0.03),) * 2, rng=rng)
    sw = scene.serpentine((-half, half), (-half - 0.06, half + 0.06), 0.02, 0.5, 61.0, 0.10, rng)
    soil_est = dataclasses.replace(soil, eps=soil.eps * rng.uniform(0.9, 1.1), alpha=soil.alpha * rng.uniform(0.8, 1.2))
    grid = sar.Grid(np.arange(-half, half + 1e-3, 0.01), np.arange(-half, half + 1e-3, 0.01), np.arange(0.02, 0.351, 0.01))
    tr = uwb_sim.simulate(sc.scatterers(soil), sw.pos_true, soil, rng=rng, noise_db=-41)
    a = uwb_proc.preprocess(tr, sw.pos_est, soil_est, band=HIGH_BAND, svd_gate_ns=1.5)
    del tr
    E = features.smooth_map(fusion.energy_map(sar.backproject(a, sw.pos_est, soil_est, grid)), grid, 0.015)
    X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
    pk = fusion.peak_index(E, grid, (0.0, 0.0), r_obj=max(0.08, R))
    shp = features.footprint_shape(E, grid, pk)
    prof = features.column_profile(a, sw.pos_est, soil_est, shp["center"], ZF)
    bg = np.hypot(X, Y) > max(0.10, R + 0.06)
    for t in targets[1:]:
        bg &= np.hypot(X - t.x, Y - t.y) > 0.06
    noise = np.sqrt(np.mean(E[bg] ** 2))

    def feats(shp, prof):
        lay = features.echo_layers(prof, ZF, top_rel_db=-6.0)
        return dict(aspect=shp["aspect"], major_m=shp["major_m"], minor_m=shp["minor_m"], center=shp["center"],
                    top_z=lay["top_z"], bottom_z=lay["bottom_z"], n_peaks=lay["n_peaks"],
                    thick_ratio=(lay["bottom_z"] - lay["top_z"]) / max(shp["major_m"], 1e-3),
                    bottom_top_db=float(20 * np.log10(lay["bottom_amp"] / lay["top_amp"])),
                    spec_ratio=features.spectral_centroid_ratio(prof, ZF, lay["top_z"], soil_est, band=HIGH_BAND))

    f = dict(snr_db=float(20 * np.log10(E[pk] / noise)), **feats(shp, prof))
    d1 = np.hypot(X - X[pk], Y - Y[pk])
    cand = (np.hypot(X, Y) <= max(0.08, R)) & (d1 >= 0.05) & (E >= maximum_filter(E, size=3))
    alt = None
    if cand.any():
        pk2 = np.unravel_index(np.argmax(np.where(cand, E, -np.inf)), E.shape)
        snr2 = float(20 * np.log10(E[pk2] / noise))
        if snr2 >= 6.0:
            E2 = np.where(np.hypot(X - X[pk2], Y - Y[pk2]) <= d1, E, 0.0)
            shp2 = features.footprint_shape(E2, grid, pk2)
            prof2 = features.column_profile(a, sw.pos_est, soil_est, shp2["center"], ZF)
            alt = dict(snr_db=snr2, **feats(shp2, prof2))
    del a
    row = dict(i=i, set=set_name, soil=sname, kind=kind, cls="mine" if kind in MINES else ("rock" if kind == "rock" else "other"),
               depth=depth, yaw=float(getattr(obj, "yaw", 0.0)), hi_gate={"new": f},
               dwell10=dwell(soil, targets, shp["center"], base + 7000 + i, 10),
               dwell40=dwell(soil, targets, shp["center"], base + 8000 + i, 40),
               dwell80=dwell(soil, targets, shp["center"], base + 9000 + i, 80),
               dwell160=dwell(soil, targets, shp["center"], base + 9500 + i, 160))
    if set_name in ("frag_train", "test6"):
        row["targets"] = [dict(name=t.name, x=float(t.x), y=float(t.y), depth=float(t.depth)) for t in targets]
        row["alt"] = alt
        if alt is not None:
            c2 = alt["center"]
            row.update(alt_dwell10=dwell(soil, targets, c2, base + 13000 + i, 10),
                       alt_dwell40=dwell(soil, targets, c2, base + 14000 + i, 40),
                       alt_dwell80=dwell(soil, targets, c2, base + 15000 + i, 80),
                       alt_dwell160=dwell(soil, targets, c2, base + 15500 + i, 160))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS), required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = set_size(args.set)
    todo = [int(x) for x in args.only.split(",")] if args.only else list(range(n))[args.shard::args.nshards]
    with Pool(args.workers) as pool, open(args.out, "a") as fh:
        for k, row in enumerate(pool.imap_unordered(one, [(i, args.set) for i in todo]), 1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"{k}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
