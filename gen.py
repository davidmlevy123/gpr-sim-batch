import argparse
import dataclasses
import json
from multiprocessing import Pool

import numpy as np

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
SETS = {"train": (500000, 3), "test": (600000, 2), "test2": (700000, 2)}
ZF = np.arange(0.01, 0.40, 0.001)
MINES = {"vs50", "pmn2", "tm62p3"}


def make_object(kind, rng, depth):
    x, y = rng.uniform(-0.03, 0.03, 2)
    yaw = rng.uniform(0, np.pi)
    if kind == "vs50":
        return scene.vs50(x, y, depth)
    if kind == "pmn2":
        return scene.pmn2(x, y, depth)
    if kind == "tm62p3":
        return scene.tm62p3(x, y, depth)
    if kind == "rock":
        a = rng.uniform(0.03, 0.06)
        return scene.Ellipsoid("rock", x, y, depth, a, a * rng.uniform(0.7, 1.0), a * rng.uniform(0.4, 0.7),
                               rng.uniform(4.5, 7.5), yaw=yaw)
    if kind == "water_bottle":
        return scene.Ellipsoid("water bottle", x, y, depth, 0.10, 0.035, 0.035, 80.0, compliance=0.8, yaw=yaw)
    if kind == "empty_bottle":
        return scene.Ellipsoid("empty bottle", x, y, depth, 0.10, 0.035, 0.035, 1.05,
                               f0=rng.uniform(250, 600), q=4.0, gain=4.0, yaw=yaw)
    if kind == "can":
        return scene.Ellipsoid("crushed can", x, y, depth, 0.05, 0.035, 0.02, 1e4, compliance=0.9, yaw=yaw)
    return scene.Ellipsoid("wet root", x, y, depth, 0.10, 0.015, 0.015, 20.0, compliance=0.9, yaw=yaw)


def dwell(soil, obj, xy, seed, n):
    sim = vibro.VibroSim(soil, np.random.default_rng(seed), phase_noise=1e-4, oversample=10)
    w = vibro.calibrate(sim, n_chirps=3)
    s = vibro.dwell_score(sim, w, [obj], xy, n_chirps=n)
    ref = np.median([vibro.dwell_score(sim, w, [], (0.5, 0.5), n_chirps=n) for _ in range(3)])
    return float(s / ref)


def one(args):
    i, set_name = args
    base, _ = SETS[set_name]
    s, kind, (d0, d1) = CELLS[i % len(CELLS)]
    sname, soil0 = SOILS[s]
    soil = dataclasses.replace(soil0, alpha_exp=1.0)
    rng = np.random.default_rng(base + i)
    depth = float(rng.uniform(d0, d1))
    obj = make_object(kind, rng, depth)
    R = max(getattr(obj, "footprint_radius", 0.05), getattr(obj, "a", 0.0))
    half = max(0.12, R + 0.08)
    sc = scene.Scene([obj], n_pebbles=10, extent=((-half - 0.03, half + 0.03),) * 2, rng=rng)
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
    del a
    lay = features.echo_layers(prof, ZF, top_rel_db=-6.0)
    f = dict(snr_db=float(20 * np.log10(E[pk] / np.sqrt(np.mean(E[np.hypot(X, Y) > max(0.10, R + 0.06)] ** 2)))),
             aspect=shp["aspect"], major_m=shp["major_m"], minor_m=shp["minor_m"], center=shp["center"],
             top_z=lay["top_z"], bottom_z=lay["bottom_z"], n_peaks=lay["n_peaks"],
             thick_ratio=(lay["bottom_z"] - lay["top_z"]) / max(shp["major_m"], 1e-3),
             bottom_top_db=float(20 * np.log10(lay["bottom_amp"] / lay["top_amp"])),
             spec_ratio=features.spectral_centroid_ratio(prof, ZF, lay["top_z"], soil_est, band=HIGH_BAND))
    return dict(i=i, set=set_name, soil=sname, kind=kind, cls="mine" if kind in MINES else ("rock" if kind == "rock" else "other"),
                depth=depth, yaw=float(getattr(obj, "yaw", 0.0)), hi_gate={"new": f},
                dwell10=dwell(soil, obj, shp["center"], base + 7000 + i, 10),
                dwell40=dwell(soil, obj, shp["center"], base + 8000 + i, 40))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS), required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = len(CELLS) * SETS[args.set][1]
    todo = [int(x) for x in args.only.split(",")] if args.only else list(range(n))[args.shard::args.nshards]
    with Pool(args.workers) as pool, open(args.out, "a") as fh:
        for k, row in enumerate(pool.imap_unordered(one, [(i, args.set) for i in todo]), 1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"{k}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
