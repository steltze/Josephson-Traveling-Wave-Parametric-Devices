"""
Programmatic launcher for the S-parameter dashboard.

Run your simulation(s) exactly as before, then hand the resulting
S-matrices straight to `Dashboard` — nothing gets re-run inside the
dashboard, it only lets you pick which S_ij (from which run) to plot,
in dB, without re-cascading anything.

Example
-------
>>> from simulation import SimulationConfig, Simulation
>>> from models import JTLDiscrete
>>> from dashboard import Dashboard
>>>
>>> cfg1 = SimulationConfig(epsilon=0.05)
>>> cfg2 = SimulationConfig(epsilon=0.10)
>>> S1 = Simulation(JTLDiscrete, cfg1).get_s_matrix()
>>> S2 = Simulation(JTLDiscrete, cfg2).get_s_matrix()
>>>
>>> Dashboard([S1, S2], freqs=[cfg1.freqs, cfg2.freqs], labels=["eps=0.05", "eps=0.10"]).run()

If every run shares the same frequency grid, pass `freqs` once instead of
a list — `Dashboard([S1, S2], freqs=cfg.freqs)`.

Pass `ks_state` (the same list used to build `SimulationConfig`) to get
ports labelled by physics ("left signal", "right idler (k=+1)") instead of
raw port indices ("S13") — see `port_labels_from_ks_state`. For a
multi-pump run (`models.jtl_discrete_multipump.JTLDiscreteMultiPump`), pass
`omega_pump`/`Kmax` instead (the same values used to build
`SimulationConfig`) — see `port_labels_from_multipump`.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_APP_PATH = Path(__file__).resolve().parent / "app.py"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _free_port(port: int) -> None:
    """
    Kill any leftover process already bound to `port`.

    Streamlit only auto-picks a different port when none was requested;
    since `run()` always passes `--server.port` explicitly, a stale
    dashboard process still holding the port (e.g. the previous run wasn't
    Ctrl+C'd) makes the next launch fail outright instead of reusing the
    port. Only processes whose command line mentions "streamlit" are
    killed, so an unrelated program on the same port is left alone.
    """
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True
        )
    except FileNotFoundError:
        return  # lsof not available; best-effort only
    pids = result.stdout.split()

    for pid in pids:
        cmdline = subprocess.run(
            ["ps", "-p", pid, "-o", "command="], capture_output=True, text=True
        ).stdout
        if "streamlit" not in cmdline:
            continue
        with contextlib.suppress(ProcessLookupError):
            os.kill(int(pid), signal.SIGTERM)

    if pids:
        time.sleep(0.5)


def _venv_python() -> str:
    """
    Path to this project's `.venv` interpreter, where the dashboard's
    dependencies (streamlit, ...) are installed.

    Falls back to `sys.executable` if no `.venv` is found next to the
    project root (e.g. a differently-named/located virtual environment) --
    but prefers the project `.venv` so `Dashboard.run()` still works even
    when the calling script was invoked with the system interpreter instead
    of the venv's (a symlinked venv `python3` can make `sys.executable`
    resolve to the system binary rather than the venv one).
    """
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    exe_name = "python.exe" if os.name == "nt" else "python"
    candidate = _PROJECT_ROOT / ".venv" / bin_dir / exe_name
    return str(candidate) if candidate.exists() else sys.executable


def port_labels_from_ks_state(ks_state: list[int]) -> list[str]:
    """
    Physical port names for the standard single-pump port ordering: the
    state vector is `ks_state` on the left (input) side followed by the
    same `ks_state` on the right (output/transmitted) side (see
    `Simulation.get_s_matrix` / `julia_comparison.py`'s
    `central_band`/`transmitted_band = central_band + len(ks_state)`).

    k=0 is the signal; k!=0 is an idler at that many pump harmonics away.
    """
    names = ["signal" if k == 0 else f"idler (k={k:+d})" for k in ks_state]
    return [f"{n} left" for n in names] + [f"{n} right" for n in names]


def port_labels_from_multipump(
    omega_pump: list[float], Kmax: list[tuple[int, int]]
) -> list[str]:
    """
    Physical port names for a multi-pump lattice (see
    `models.jtl_discrete_multipump.JTLDiscreteMultiPump`): the state vector
    is the tensor lattice of per-pump sideband indices (k_1, ..., k_P) --
    `models.electrical_elements.multipump_frequency_grid` gives that
    lattice's ordering -- on the left (input) side, followed by the same
    lattice on the right (output) side (see `examples/multi_pump_jtl_example.py`,
    which indexes the transmitted port for lattice state `p` at `D + p`).

    The all-zero state is the signal; any other state is an idler offset by
    that many harmonics of each pump (pump index is 1-based, e.g.
    `idler (p1=-1)` is one harmonic below pump 1).
    """
    from models.electrical_elements import multipump_frequency_grid

    _, lattice_labels = multipump_frequency_grid(0.0, omega_pump, Kmax)
    names = []
    for state in lattice_labels:
        if all(k == 0 for k in state):
            names.append("signal")
        else:
            offsets = ", ".join(
                f"p{p + 1}={k:+d}" for p, k in enumerate(state) if k != 0
            )
            names.append(f"idler ({offsets})")
    return [f"{n} left" for n in names] + [f"{n} right" for n in names]


class Dashboard:
    """
    Collects one or more S-matrices (+ their frequency grids) and launches
    the Streamlit S-parameter viewer on them via `run()`.
    """

    def __init__(
        self,
        s_matrices,
        freqs,
        labels: list[str] | None = None,
        ks_state: list[int] | list[list[int]] | None = None,
        omega_pump: list[float] | None = None,
        Kmax: list[tuple[int, int]] | None = None,
        port_labels: list[str] | list[list[str]] | None = None,
    ):
        s_matrices = [
            np.asarray(getattr(S, "array", S), dtype=complex) for S in s_matrices
        ]
        n = len(s_matrices)

        if isinstance(freqs, (list, tuple)):
            if len(freqs) != n:
                raise ValueError(f"Got {n} S-matrices but {len(freqs)} freq arrays")
            freqs_list = [np.asarray(f, dtype=float) for f in freqs]
        else:
            freqs_list = [np.asarray(freqs, dtype=float)] * n

        for S, f in zip(s_matrices, freqs_list):
            if S.ndim != 3 or S.shape[1] != S.shape[2]:
                raise ValueError(f"Each S-matrix must be (Nf, N, N), got {S.shape}")
            if f.shape[0] != S.shape[0]:
                raise ValueError(
                    f"freqs length {f.shape[0]} does not match Nf={S.shape[0]}"
                )

        labels = (
            list(labels) if labels is not None else [f"run{i + 1}" for i in range(n)]
        )
        if len(labels) != n:
            raise ValueError(f"Got {n} S-matrices but {len(labels)} labels")

        # Port labels, in priority order: explicit `port_labels` > derived
        # from `ks_state` (single-pump) > derived from `omega_pump`+`Kmax`
        # (multi-pump, shared by every run) > a generic "port i" fallback.
        # `port_labels`/`ks_state` may be a single list shared by every run,
        # or one list per run.
        if port_labels is not None:
            if len(port_labels) == n and all(
                isinstance(x, (list, tuple)) for x in port_labels
            ):
                port_labels_list = [list(x) for x in port_labels]
            else:
                port_labels_list = [list(port_labels)] * n
        elif ks_state is not None:
            if len(ks_state) == n and all(
                isinstance(x, (list, tuple)) for x in ks_state
            ):
                ks_state_list = [list(x) for x in ks_state]
            else:
                ks_state_list = [list(ks_state)] * n
            port_labels_list = [port_labels_from_ks_state(ks) for ks in ks_state_list]
        elif omega_pump is not None or Kmax is not None:
            if omega_pump is None or Kmax is None:
                raise ValueError("`omega_pump` and `Kmax` must be given together")
            port_labels_list = [port_labels_from_multipump(omega_pump, Kmax)] * n
        else:
            port_labels_list = [
                [f"port {p + 1}" for p in range(S.shape[1])] for S in s_matrices
            ]

        for S, pl in zip(s_matrices, port_labels_list):
            if len(pl) != S.shape[1]:
                raise ValueError(
                    f"Got {len(pl)} port labels but S-matrix has {S.shape[1]} ports"
                )

        self.runs = [
            {"S": S, "freqs": f, "label": label, "port_labels": pl}
            for S, f, label, pl in zip(s_matrices, freqs_list, labels, port_labels_list)
        ]

    def run(self, port: int = 8501) -> None:
        """
        Serialize the runs to a temp file and launch `streamlit run` on
        them. Blocks until the dashboard process is stopped (Ctrl+C),
        like `plt.show()`.

        Runs headless (no auto-opened browser tab): since every call reuses
        the same fixed `port` (freeing it from a leftover run if needed),
        leaving one browser tab open at that URL across repeated
        Ctrl+C/rerun cycles just reconnects it to the new run instead of
        piling up a new tab each time.
        """
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(self.runs, f)
            data_path = f.name

        _free_port(port)

        print(f"Dashboard: http://localhost:{port}  (leave this tab open; reruns reuse it)")

        cmd = [
            _venv_python(),
            "-m",
            "streamlit",
            "run",
            str(_APP_PATH),
            "--server.address",
            "localhost",
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--",
            "--data",
            data_path,
        ]
        try:
            subprocess.run(cmd)
        finally:
            Path(data_path).unlink(missing_ok=True)
