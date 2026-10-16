"""Проведение серий независимых запусков и запись результатов в CSV."""

import csv
import os
import statistics
import time


def run_series(runner, runs=21, base_seed=0, **kwargs):
    """Серия из runs независимых запусков с разными seed."""
    results = []
    for i in range(runs):
        seed = base_seed + i
        t0 = time.perf_counter()
        res = runner(seed=seed, **kwargs)
        res["seed"] = seed
        res["wall_time"] = time.perf_counter() - t0
        results.append(res)
    return results, summarize(results)


def summarize(results):
    finals = [r["best_value"] for r in results]
    rooms = [r.get("best_rooms", 0) for r in results]
    times = [r.get("wall_time", 0.0) for r in results]
    fes = [r.get("fe", 0) for r in results]
    frac = [r.get("feasible_frac", 1.0) for r in results]
    return {
        "runs": len(results),
        "best": max(finals),
        "worst": min(finals),
        "mean": statistics.fmean(finals),
        "median": statistics.median(finals),
        "stdev": statistics.stdev(finals) if len(finals) > 1 else 0.0,
        "mean_rooms": statistics.fmean(rooms),
        "mean_time_s": statistics.fmean(times),
        "mean_fe": statistics.fmean(fes),
        "mean_feasible_frac": statistics.fmean(frac),
        "finals": finals,
    }


def trajectory_grid(results, n_points=200):
    max_fe = max(r["history"][-1][0] for r in results)
    grid = [int(round(max_fe * i / (n_points - 1))) for i in range(n_points)]

    def interp(hist, g):
        xs = [h[0] for h in hist]
        ys = [h[1] for h in hist]
        out = []
        for target in g:
            if target <= xs[0]:
                out.append(ys[0])
                continue
            if target >= xs[-1]:
                out.append(ys[-1])
                continue
            j = 0
            while xs[j + 1] < target:
                j += 1
            x0, y0 = xs[j], ys[j]
            x1, y1 = xs[j + 1], ys[j + 1]
            out.append(y0 + (y1 - y0) * (target - x0) / (x1 - x0)
                       if x1 != x0 else y0)
        return out

    return grid, [interp(r["history"], grid) for r in results]


def curve_stats(results, n_points=200):
    grid, curves = trajectory_grid(results, n_points)
    mins = [min(c[i] for c in curves) for i in range(len(grid))]
    means = [statistics.fmean([c[i] for c in curves]) for i in range(len(grid))]
    maxs = [max(c[i] for c in curves) for i in range(len(grid))]
    return grid, mins, means, maxs


def write_runs_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "best_value", "best_rooms", "best_waste",
                         "fe", "generations", "feasible_frac", "wall_time_s"])
        for r in results:
            writer.writerow([
                r["seed"], r["best_value"], r.get("best_rooms", ""),
                r.get("best_waste", ""), r.get("fe", 0),
                r.get("generations", 0),
                f"{r.get('feasible_frac', 1.0):.4f}",
                f"{r.get('wall_time', 0.0):.4f}",
            ])


def write_history_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "fe", "best_value"])
        for r in results:
            for fe, best in r["history"]:
                writer.writerow([r["seed"], fe, best])


def append_summary_csv(stats, name, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["config", "runs", "best", "mean", "median", "stdev", "worst",
              "mean_rooms", "mean_time_s", "mean_fe", "mean_feasible_frac"]
    row = [name, stats["runs"], stats["best"], f"{stats['mean']:.2f}",
           f"{stats['median']:.2f}", f"{stats['stdev']:.2f}", stats["worst"],
           f"{stats['mean_rooms']:.2f}",
           f"{stats['mean_time_s']:.4f}", f"{stats['mean_fe']:.1f}",
           f"{stats['mean_feasible_frac']:.4f}"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
