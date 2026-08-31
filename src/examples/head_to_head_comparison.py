import csv
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from utils import COLOR_JULIA, COLOR_PYTHON, PAPER_STYLE

from logger import get_logger, setup_logging
from simulation import Simulation

log = get_logger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
JULIA_SCRIPT = os.path.join(REPO_ROOT, "julia", "josephsoncircuits_comparison.jl")
JULIA_CSV = os.path.join(FIGURES_DIR, "julia_sparams.csv")


JULIA_TIMINGS_CSV = os.path.join(FIGURES_DIR, "julia_timings.csv")

# Repeats per simulator for the averaged benchmark: run 1 is "cold" (for
# Julia, includes JIT-compiling hbsolve's specializations; the numpy backend
# on the Python side has no such compile step), runs 2+ are "warm" and are
# what gets averaged for the reported speedup. Each extra repeat re-pays a
# full solve, so keep this low on modest hardware; override with
# TWPD_BENCH_REPEATS.
BENCH_REPEATS = int(os.environ.get("TWPD_BENCH_REPEATS", "2"))


def run_python(
    repeats: int,
    *,
    M: int = 1,
    ks_state=None,
    ncell: int = 320,
    n_freqs: int = 500,
    backend: str = "numba",
):
    """Run julia_comparison.py's config `repeats` times.

    `M`/`ks_state`/`ncell`/`n_freqs` mirror Julia's Npumpharmonics/
    Nmodulationharmonics harmonic order, cell count, and frequency
    resolution -- override them to benchmark a different problem size.

    Returns (freqs, S_params, ks_state, seconds) where `seconds` is a list of
    per-run times for `Simulation(...)` construction + `get_s_matrix(...)`
    only -- module import, the correctness checks (determinant/photon-flux),
    dispersion-relation plotting, and figure saving that julia_comparison()
    also does are excluded, since none of those are comparable to Julia's
    hbsolve-only timing.

    `backend` defaults to "numba" (@njit(cache=True)) rather than this
    project's production "numpy" backend, so run 1 pays a JIT-compilation
    cost like Julia's hbsolve does -- making run 1 vs runs 2+ a real
    cold/warm split instead of numpy's uniformly-warm timings. Caveat: numba
    caches compiled code to *disk*, keyed by argument shapes/dtypes, so
    "cold" here only means "first repeat in this call to run_python" -- if
    this (ncell, M, ks_state) combination was already compiled in an earlier
    process on this machine, run 1 loads from that cache and is warm too.
    Julia recompiles fresh every subprocess launch, so its cold number is
    reliably cold; numba's isn't, unless this is genuinely a fresh cache.
    """
    import julia_comparison as jc

    captured = {}
    timing = {}
    real_init = Simulation.__init__
    real_get_s_matrix = Simulation.get_s_matrix

    def timed_init(self, *args, **kwargs):
        timing["start"] = time.perf_counter()
        return real_init(self, *args, **kwargs)

    def capturing_get_s_matrix(self, *args, **kwargs):
        result = real_get_s_matrix(self, *args, **kwargs)
        timing["stop"] = time.perf_counter()
        captured["S_params"] = result.array
        captured["freqs"] = self._cfg.freqs
        captured["ks_state"] = self._cfg.ks_state
        return result

    Simulation.__init__ = timed_init
    Simulation.get_s_matrix = capturing_get_s_matrix
    seconds = []
    try:
        for rep in range(1, repeats + 1):
            jc.julia_comparison(
                dashboard=False, M=M, ks_state=ks_state, ncell=ncell,
                n_freqs=n_freqs, backend=backend,
            )
            elapsed = timing["stop"] - timing["start"]
            seconds.append(elapsed)
            log.info("  run %d/%d: %.3fs", rep, repeats, elapsed)
    finally:
        Simulation.__init__ = real_init
        Simulation.get_s_matrix = real_get_s_matrix

    return captured["freqs"], captured["S_params"], captured["ks_state"], seconds


def run_julia(
    skip: bool,
    repeats: int,
    *,
    ncell: int = 320,
    n_freqs: int = 500,
    nmod_harmonics: int = 1,
    npump_harmonics: int = 1,
):
    """Return (freqs, S21, S31, S11, seconds).

    `ncell`/`n_freqs`/`nmod_harmonics`/`npump_harmonics` mirror the Julia
    script's nr_cells/n_freqs/Nmodulationharmonics/Npumpharmonics -- override
    them to benchmark a different problem size.

    `seconds` is the list of per-run hbsolve times written by the Julia
    script itself to `julia_timings.csv` (run 1 cold, runs 2+ warm) -- this
    excludes Julia startup, package precompilation, and plotting, which are
    not comparable to `python_seconds`. It's `None` when `skip` reuses
    results from a previous run and no timings file exists yet.
    """
    if skip:
        if not os.path.isfile(JULIA_CSV):
            raise FileNotFoundError(
                f"{JULIA_CSV} doesn't exist yet -- can't --skip-julia on a first run."
            )
        log.info("Skipping Julia run, reusing existing %s", JULIA_CSV)
    else:
        log.info(
            "Running %s (this can take a while, %d repeats)...",
            JULIA_SCRIPT, repeats,
        )
        env = dict(
            os.environ,
            TWPD_BENCH_REPEATS=str(repeats),
            TWPD_NR_CELLS=str(ncell),
            TWPD_N_FREQS=str(n_freqs),
            TWPD_NMOD_HARMONICS=str(nmod_harmonics),
            TWPD_NPUMP_HARMONICS=str(npump_harmonics),
        )
        subprocess.run(
            ["julia", "--threads=8", JULIA_SCRIPT], cwd=REPO_ROOT, check=True, env=env
        )

    julia_times = None
    if os.path.isfile(JULIA_TIMINGS_CSV):
        julia_times = []
        with open(JULIA_TIMINGS_CSV) as f:
            for row in csv.DictReader(f):
                julia_times.append(float(row["seconds"]))

    freq_jl, S21_jl, S31_jl, S11_jl = [], [], [], []
    with open(JULIA_CSV) as f:
        for row in csv.DictReader(f):
            freq_jl.append(float(row["freq_GHz"]))
            S21_jl.append(complex(float(row["S21_re"]), float(row["S21_im"])))
            S31_jl.append(complex(float(row["S31_re"]), float(row["S31_im"])))
            S11_jl.append(complex(float(row["S11_re"]), float(row["S11_im"])))
    return (
        np.array(freq_jl),
        np.array(S21_jl),
        np.array(S31_jl),
        np.array(S11_jl),
        julia_times,
    )


def cold_warm(times):
    """(cold, warm_mean) from a per-run timing list; warm_mean is None if len < 2."""
    warm = times[1:]
    return times[0], (sum(warm) / len(warm) if warm else None)


# (label, Python ks_state/M, Julia Nmodulationharmonics/Npumpharmonics) --
# ks_state and the harmonic orders grow together since they track the same
# physics: more sidebands tracked in the linear response (ks_state) needs a
# matching Floquet truncation order (M / Nmodulationharmonics/Npumpharmonics)
# for the pump's own harmonic-balance solve.
SWEEP_CONFIGS = [
    {"label": "M=1, 3 bands", "ks_state": [-1, 0, 1], "M": 1, "nmod": 1, "npump": 1},
    {"label": "M=2, 5 bands", "ks_state": [-2, -1, 0, 1, 2], "M": 2, "nmod": 2, "npump": 2},
    {
        "label": "M=3, 7 bands",
        "ks_state": [-3, -2, -1, 0, 1, 2, 3],
        "M": 3,
        "nmod": 3,
        "npump": 3,
    },
    {
        "label": "M=4, 8 bands",
        "ks_state": [-4, -3, -2, -1, 0, 1, 2, 3, 4],
        "M": 4,
        "nmod": 4,
        "npump": 4,
    },
        {
        "label": "M=5, 10 bands",
        "ks_state": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        "M": 5,
        "nmod": 5,
        "npump": 5,
    },
]


def run_sweep(repeats: int, ncell: int, n_freqs: int):
    """Benchmark Python vs Julia across SWEEP_CONFIGS at a shared (ncell, n_freqs).

    Unlike main(), this doesn't produce the S-parameter correctness plot --
    it's purely a timing sweep, so --skip-julia isn't supported (each config
    needs its own fresh Julia run; there's nothing to cache per-config).
    """
    setup_logging()
    log.info(
        "Sweeping %d configs at ncell=%d, n_freqs=%d, %d repeats each...",
        len(SWEEP_CONFIGS), ncell, n_freqs, repeats,
    )

    rows = []
    for cfg in SWEEP_CONFIGS:
        log.info("--- %s ---", cfg["label"])
        _, _, _, python_seconds = run_python(
            repeats, M=cfg["M"], ks_state=cfg["ks_state"], ncell=ncell, n_freqs=n_freqs
        )
        _, _, _, _, julia_seconds = run_julia(
            skip=False,
            repeats=repeats,
            ncell=ncell,
            n_freqs=n_freqs,
            nmod_harmonics=cfg["nmod"],
            npump_harmonics=cfg["npump"],
        )
        py_cold, py_warm = cold_warm(python_seconds)
        jl_cold, jl_warm = cold_warm(julia_seconds)
        rows.append((cfg["label"], py_cold, py_warm, jl_cold, jl_warm))

    log.info("")
    log.info(
        "%-14s %10s %10s %10s %10s %10s",
        "config", "py cold", "py warm", "jl cold", "jl warm", "speedup",
    )
    warm_speedups = []
    for label, py_cold, py_warm, jl_cold, jl_warm in rows:
        if py_warm is not None and jl_warm is not None:
            speedup = jl_warm / py_warm
            warm_speedups.append(speedup)
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "n/a"
        log.info(
            "%-14s %9.3fs %9.3fs %9.3fs %9.3fs %10s",
            label, py_cold, py_warm if py_warm is not None else float("nan"),
            jl_cold, jl_warm if jl_warm is not None else float("nan"), speedup_str,
        )

    if warm_speedups:
        # Geometric mean: the right average for a set of ratios -- it's
        # invariant to which side (Python/Julia) each speedup is expressed
        # relative to, unlike an arithmetic mean of the same ratios.
        log_mean = sum(np.log(s) for s in warm_speedups) / len(warm_speedups)
        geo_mean = float(np.exp(log_mean))
        direction = "Julia" if geo_mean >= 1 else "Python"
        log.info(
            "-> Average speedup across %d configs (geometric mean): "
            "%s is %.2fx faster (Julia/Python warm-time ratios: %s)",
            len(warm_speedups), direction,
            geo_mean if geo_mean >= 1 else 1 / geo_mean,
            ", ".join(f"{s:.2f}xs" for s in warm_speedups),
        )


def main(skip_julia: bool, repeats: int = BENCH_REPEATS, python_backend: str = "numba"):
    setup_logging()

    log.info(
        "Running Python simulator (julia_comparison.py's config, %d repeats, "
        "backend=%s)...",
        repeats, python_backend,
    )
    freqs_py, S_py, ks_state, python_seconds = run_python(repeats, backend=python_backend)
    if ks_state != [0, 1]:
        log.warning(
            "julia_comparison.py's ks_state is %s, not [0, 1] -- the S21/S31/S11 "
            "port indices below assume [0, 1] and may be wrong.",
            ks_state,
        )
    dB = lambda x: 20 * np.log10(np.abs(x) + 1e-30)
    S21_py = S_py[:, 2, 0]  # right-signal <- left-signal: direct transmission
    S31_py = S_py[:, 1, 0]  # right-idler  <- left-signal: TWPC conversion
    S11_py = S_py[:, 0, 0]  # left-signal  <- left-signal: input return loss

    freqs_jl, S21_jl, S31_jl, S11_jl, julia_seconds = run_julia(skip_julia, repeats)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    with mpl.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(
            2, 1, figsize=(3.4, 4.6), sharex=True,
            gridspec_kw={"height_ratios": [1.15, 1]},
        )

        for ax, (S_p, S_j, title) in zip(
            axes,
            [
                (S21_py, S21_jl, "Direct transmission (signal → signal)"),
                (S31_py, S31_jl, "Conversion (signal → idler)"),
            ],
        ):
            ax.plot(
                freqs_py, dB(S_p), label="Python (transfer-matrix)",
                color=COLOR_PYTHON, lw=1.5, zorder=2, solid_capstyle="round",
            )
            ax.plot(
                freqs_jl, dB(S_j), label="Julia (JosephsonCircuits.jl)",
                color=COLOR_JULIA, lw=1.3, zorder=3, ls=(0, (1, 1.4)),
                dash_capstyle="round",
            )
            ax.set_ylabel("|S| (dB)")
            ax.set_title(title, loc="left")
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0].set_xlim(1, 8)
        axes[-1].set_xlabel("Signal frequency (GHz)")
        fig.align_ylabels(axes)

        # Figure-level legend above both panels -- keeps it clear of the
        # data instead of fighting for empty space inside an axes.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.tight_layout(h_pad=1.2, rect=(0, 0, 1, 0.93))
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.56, 1.0),
            ncol=1, frameon=False, handlelength=2.6,
        )

        svg_path = os.path.join(FIGURES_DIR, "head_to_head_comparison.svg")
        fig.savefig(svg_path)
        log.info("Saved %s", svg_path)

        png_path = os.path.join(FIGURES_DIR, "head_to_head_comparison.png")
        fig.savefig(png_path, dpi=300)
        log.info("Saved %s", png_path)

    i_py = np.argmax(dB(S31_py))
    i_jl = np.argmax(dB(S31_jl))
    log.info(
        "Python S31 peak: %.2f dB at %.3f GHz", dB(S31_py)[i_py], freqs_py[i_py]
    )
    log.info(
        "Julia  S31 peak: %.2f dB at %.3f GHz", dB(S31_jl)[i_jl], freqs_jl[i_jl]
    )

    py_cold, py_warm = cold_warm(python_seconds)
    log.info(
        "Python simulator: run 1 (cold) %.3fs, %s -- all runs: %s",
        py_cold,
        f"warm mean {py_warm:.3f}s" if py_warm is not None else "no warm runs",
        ", ".join(f"{t:.3f}s" for t in python_seconds),
    )

    if julia_seconds is None:
        log.info("Julia simulator:  skipped, no fresh timing (--skip-julia)")
    else:
        jl_cold, jl_warm = cold_warm(julia_seconds)
        log.info(
            "Julia simulator:  run 1 (cold) %.3fs, %s -- all runs: %s",
            jl_cold,
            f"warm mean {jl_warm:.3f}s" if jl_warm is not None else "no warm runs",
            ", ".join(f"{t:.3f}s" for t in julia_seconds),
        )

        # Steady-state ("warm") speedup is the fair number for repeated use;
        # cold-start is reported separately since it's dominated by Julia's
        # one-time JIT compilation, not solver performance.
        if py_warm is not None and jl_warm is not None:
            ratio = jl_warm / py_warm
            direction = "Python" if ratio >= 1 else "Julia"
            log.info(
                "-> Average (warm) speedup: %s simulator is %.1fx faster",
                direction, ratio if ratio >= 1 else 1 / ratio,
            )
        else:
            log.info(
                "-> No warm runs on both sides (repeats=%d) -- can't compute an "
                "average speedup; increase --repeats.",
                repeats,
            )

        cold_ratio = jl_cold / py_cold
        cold_direction = "Python" if cold_ratio >= 1 else "Julia"
        log.info(
            "-> Cold-start (incl. Julia JIT compile): %s simulator is %.1fx faster",
            cold_direction, cold_ratio if cold_ratio >= 1 else 1 / cold_ratio,
        )


if __name__ == "__main__":
    repeats = BENCH_REPEATS
    sweep_ncell, sweep_n_freqs = 25, 25
    for arg in sys.argv[1:]:
        if arg.startswith("--repeats="):
            repeats = int(arg.split("=", 1)[1])
        elif arg.startswith("--sweep-ncell="):
            sweep_ncell = int(arg.split("=", 1)[1])
        elif arg.startswith("--sweep-n-freqs="):
            sweep_n_freqs = int(arg.split("=", 1)[1])

    if "--sweep" in sys.argv:
        run_sweep(repeats, sweep_ncell, sweep_n_freqs)
    else:
        main(skip_julia="--skip-julia" in sys.argv, repeats=repeats)
