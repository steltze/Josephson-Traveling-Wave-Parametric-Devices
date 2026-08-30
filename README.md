# Traveling Wave Parametric Devices

Simulation and analysis toolkit for traveling-wave Josephson parametric
amplifiers and converters (TWPA/TWPC) based on a lumped-element and discrete transmission line model using transfer-matrices to extract the network S-parameters. This software was developed in the LPENS-Quantic laboratory.

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/), which
installs the right Python version for you and keeps everyone on the exact
same package set via [`uv.lock`](uv.lock).

### 1. Install uv

**Linux / macOS**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows** (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Set up the project

Same commands on every platform — uv creates and manages `.venv` for you and
installs the exact Python version pinned in
[`.python-version`](.python-version):

```bash
git clone https://github.com/steltze/Josephson-Traveling-Wave-Parametric-Devices.git
cd Josephson-Traveling-Wave-Parametric-Devices
uv sync
```

Then run anything with `uv run`, e.g. `uv run pytest`, without manually
activating the virtual environment. If you'd rather activate it directly:

```bash
# Linux/macOS
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the activation script (`running scripts is
disabled on this system`), that's its default script-execution policy —
either run `uv run <command>` instead of activating, or allow it for the
current session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Optional extras (also installable individually, e.g. `uv sync --extra numba`):

```bash
uv sync --all-extras   # numba backend, dashboard, docs
```

- `numba` — JIT-compiled backend for the ABCD/S-matrix solvers
- `dashboard` — interactive Streamlit S-parameter viewer
- `docs` — Sphinx documentation build

### Without uv

If you'd rather not install uv, plain `pip` works too — `pyproject.toml` is
the source of truth, `uv.lock` is uv-specific:

```bash
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
pip install ".[dashboard,numba]"
```

```powershell
# Windows (PowerShell)
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install ".[dashboard,numba]"
```

## Documentation

See [`docs/getting_started.md`](docs/getting_started.md) for a quickstart
and the full API reference.
