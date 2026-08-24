# Traveling Wave Parametric Devices

Simulation and analysis toolkit for traveling-wave Josephson parametric
amplifiers and converters (TWPA/TWPC): discrete and continuous transmission
line models, transfer-matrix / ABCD-matrix solvers, dispersion-relation and
S-parameter analysis, and an interactive dashboard for inspecting results.

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/), which
installs the right Python version for you and keeps everyone on the exact
same package set via [`uv.lock`](uv.lock).

### 1. Install uv

**Linux/macOS**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows** (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

(See the [uv install docs](https://docs.astral.sh/uv/getting-started/installation/)
for other options, e.g. `pipx install uv` or `brew install uv`.)

### 2. Set up the project

Same command on every platform — uv creates and manages `.venv` for you:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
uv sync
```

Then run anything with `uv run`, e.g. `uv run pytest` or
`uv run python src/main.py`, without manually activating the virtual
environment. If you'd rather activate it directly:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\activate
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
python3 -m venv .venv          # py -m venv .venv   on Windows
source .venv/bin/activate      # .venv\Scripts\activate   on Windows
pip install ".[dashboard,numba]"
```

## Documentation

See [`docs/getting_started.md`](docs/getting_started.md) for a quickstart
and the full API reference.
