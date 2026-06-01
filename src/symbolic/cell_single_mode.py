"""
Symbolic transfer-matrix builder for TWPA/TWPC unit cells for time-periodic impedance/admittance networks.
The transfer matrix T maps the state vector [V[k,n], I[k,n]] -> [V[k,n+1], I[k,n+1]] for each sideband k simultaneously.
"""

from __future__ import annotations
from collections.abc import Callable
from pathlib import Path
import numpy as np
from sympy import (
    Add,
    Expr,
    Function,
    Indexed,
    IndexedBase,
    Matrix,
    expand,
    exp,
    symbols,
    zeros,
    I,
)
from sympy.core.function import AppliedUndef

from models.cell import CellImmitance


class CellSingleMode:
    def __init__(self) -> None:
        self.t, self.theta, self.omega_p = symbols("t theta omega_p", real=True)
        self.k, self.n = symbols("k n", integer=True)
        self.omega = IndexedBase("omega")
        self.Zs0 = Function("Zs0")
        self.Yg0 = Function("Yg0")
        self.V = IndexedBase("V")
        self.Ic = IndexedBase("I")
        self.xi = symbols("xi")
        self._lambdify_cache: dict = {}

    def _fourier_basis(self, m: int) -> Expr:
        """Return the m-th basis element exp(I*m*omega_p*t)."""
        return exp(I * m * self.omega_p * self.t)

    def _make_harmonic_functions(
        self, M: int
    ) -> tuple[list[type[Function]], list[type[Function]]]:
        """Return lists of SymPy Function objects Zs_m and Yg_m for orders 1..M."""
        Zs_m = [Function(f"Zs{m}") for m in range(1, M + 1)]
        Yg_m = [Function(f"Yg{m}") for m in range(1, M + 1)]
        return Zs_m, Yg_m

    def _build_Zs_series(self, Zs_m: list[type[Function]]) -> Expr:
        """
        Return the symbolic Zs(t) Fourier series up to order M.

        The placeholder xi is later substituted with omega[k] when extracting
        harmonic coefficients.
        """
        impedance = self.Zs0(self.xi)
        for mi, Zm in enumerate(Zs_m, start=1):
            impedance += Zm(self.xi) * self._fourier_basis(mi) * exp(
                I * mi * self.theta
            ) + Zm(self.xi) * self._fourier_basis(-mi) * exp(-I * mi * self.theta)
        return impedance

    def _build_Yg_shunt(self, Yg_m: list[type[Function]]) -> Expr:
        """Return the symbolic Yg(t) Fourier series up to order M."""
        admittance = self.Yg0(self.xi)
        for mi, Ym in enumerate(Yg_m, start=1):
            admittance += Ym(self.xi) * self._fourier_basis(mi) * exp(
                I * mi * self.theta
            ) + Ym(self.xi) * self._fourier_basis(-mi) * exp(-I * mi * self.theta)
        return admittance

    def _extract_harmonic_results(
        self, M: int, Zs_t: Expr, Yg_t: Expr
    ) -> dict[int, tuple[Expr, Expr]]:
        """
        Expand the cell update equations and extract per-harmonic coefficients.

        The cell equations are:
            V[k, n+1] = V[k, n] - Zs(t) * I[k, n]
            I[k, n+1] = I[k, n] - Yg(t) * V[k, n+1]

        Returns
        -------
        results : dict  {m: (cv, ci)}
            cv, ci are the symbolic coefficient expressions in terms of V[k, n]
            and I[k, n] after the sideband shift k -> k - m.
        """
        Vn_1 = self.V[self.k, self.n] - Zs_t * self.Ic[self.k, self.n]
        In_1 = self.Ic[self.k, self.n] - Yg_t * Vn_1

        Vn_1_expanded = expand(Vn_1)
        In_1_expanded = expand(In_1)

        ms_nonzero = [m for m in range(-2 * M, 2 * M + 1) if m != 0]
        results = {}

        for m in ms_nonzero:
            cv_raw = Vn_1_expanded.coeff(self._fourier_basis(m))
            ci_raw = In_1_expanded.coeff(self._fourier_basis(m))
            cv = cv_raw.subs(self.xi, self.omega[self.k]).subs(self.k, self.k - m)
            ci = ci_raw.subs(self.xi, self.omega[self.k]).subs(self.k, self.k - m)
            results[m] = (cv, ci)

        # DC component: subtract all oscillating parts
        V_osc = Add(
            *[
                Vn_1_expanded.coeff(self._fourier_basis(m)) * self._fourier_basis(m)
                for m in ms_nonzero
            ]
        )
        I_osc = Add(
            *[
                In_1_expanded.coeff(self._fourier_basis(m)) * self._fourier_basis(m)
                for m in ms_nonzero
            ]
        )
        cv0 = expand(Vn_1_expanded - V_osc).subs(self.xi, self.omega[self.k])
        ci0 = expand(In_1_expanded - I_osc).subs(self.xi, self.omega[self.k])
        results[0] = (cv0, ci0)

        return results

    def _build_transfer_matrix(
        self,
        M: int,
        ks_state: list[int],
        results: dict[int, tuple[Expr, Expr]],
    ) -> tuple[Matrix, list[Expr]]:
        """
        Build the symbolic transfer matrix T.

        Parameters
        ----------
        M        : truncation order (determines harmonic range ±2M)
        ks_state : list of sideband indices; each contributes a (V[k,n], I[k,n]) pair
        results  : dict from _extract_harmonic_results

        Returns
        -------
        T_sym      : sympy Matrix of shape (2*len(ks_state), 2*len(ks_state))
        state_syms : list of state symbols in row/column order
        """
        n_modes = len(ks_state)
        state_syms = [self.V[_k, self.n] for _k in ks_state] + [
            self.Ic[_k, self.n] for _k in ks_state
        ]
        dim = len(state_syms)

        ms_full = list(range(-2 * M, 2 * M + 1))
        T_sym = zeros(dim, dim)

        for row_idx, _k in enumerate(ks_state):
            cv_p = expand(Add(*[results[m][0].subs(self.k, _k) for m in ms_full]))
            ci_p = expand(Add(*[results[m][1].subs(self.k, _k) for m in ms_full]))
            for col_idx, s in enumerate(state_syms):
                T_sym[row_idx, col_idx] = cv_p.coeff(s)
                T_sym[n_modes + row_idx, col_idx] = ci_p.coeff(s)

        # The harmonic substitution subs(xi, omega[k]).subs(k, k-m) causes the
        # Yg0 frequency in the D block to be evaluated at the *source* sideband
        # frequency instead of the *row* frequency, making det(T) != 1. Fix:
        # recompute D = I + C*B from the already-correct A, B, C blocks so that
        # det(T) = det(D - C*B) = det(I) = 1 exactly.
        from sympy import eye as _eye

        B_sym = T_sym[:n_modes, n_modes:]
        C_sym = T_sym[n_modes:, :n_modes]
        T_sym[n_modes:, n_modes:] = _eye(n_modes) + C_sym * B_sym

        return T_sym, state_syms

    def build_symbolic_transfer_matrix(
        self,
        M: int,
        ks_state: list[int],
    ) -> tuple[Matrix, list[Expr], list[type[Function]], list[type[Function]]]:
        """
        Run the complete symbolic pipeline for given truncation order M and
        sideband list ks_state.

        Returns
        -------
        T_sym      : sympy Matrix
        state_syms : list of state symbols
        Zs_m       : list of Zs harmonic Function objects
        Yg_m       : list of Yg harmonic Function objects
        """
        Zs_m, Yg_m = self._make_harmonic_functions(M)
        Zs_t = self._build_Zs_series(Zs_m)
        Yg_t = self._build_Yg_shunt(Yg_m)
        results = self._extract_harmonic_results(M, Zs_t, Yg_t)
        T_sym, state_syms = self._build_transfer_matrix(M, ks_state, results)
        return T_sym, state_syms, Zs_m, Yg_m

    def _build_evaluator(self, T_sym: Matrix) -> tuple:
        """
        Compile T_sym into a fast lambdified function, cache by matrix id.

        Replaces all AppliedUndef (e.g. Zs1(omega[-2])), Indexed (e.g.
        omega[-2]), and scalar free symbols with Dummy variables, then calls
        lambdify once.  Subsequent calls with the same T_sym object skip
        compilation entirely.
        """
        from sympy import Dummy, lambdify

        applied = sorted(T_sym.atoms(AppliedUndef), key=str)
        indexed = sorted(T_sym.atoms(Indexed), key=str)
        indexed_base_names = {str(ix.base) for ix in indexed}
        scalars = sorted(
            [
                s
                for s in T_sym.free_symbols
                if not isinstance(s, IndexedBase) and str(s) not in indexed_base_names
            ],
            key=str,
        )
        keys = applied + indexed + scalars
        dummies = [Dummy() for _ in keys]
        T_replaced = T_sym.xreplace(dict(zip(keys, dummies)))
        fn = lambdify(dummies, T_replaced.tolist(), modules="numpy")
        entry = (fn, keys)
        self._lambdify_cache[id(T_sym)] = entry
        return entry

    def build_numeric_matrix(
        self,
        T_sym: Matrix,
        dim: int,
        M: int,
        ks_state: list[int],
        Zs_m: list[type[Function]],
        Yg_m: list[type[Function]],
        cell: CellImmitance,
        omega_val_fn: Callable[[int], float],
        omega_p_val: float,
        k_val: int = 0,
    ) -> np.ndarray:
        """
        Substitute numerical values into T_sym and return a numpy complex array.

        Returns
        -------
        T_num : numpy ndarray, shape (dim, dim), dtype complex
        """
        N_max = max(abs(_k) for _k in ks_state) if ks_state else 0
        k_range = range(k_val - (N_max + 2 * M), k_val + (N_max + 2 * M) + 1)

        subs = {
            self.theta: cell.theta,
            self.omega_p: omega_p_val,
        }
        for _k in k_range:
            omega_k = omega_val_fn(_k)
            subs[self.omega[_k]] = omega_k
            subs[self.Zs0(self.omega[_k])] = cell.Zs0_fn(omega_k)
            subs[self.Yg0(self.omega[_k])] = cell.Yg0_fn(omega_k)
            for mi, Zm in enumerate(Zs_m, start=1):
                subs[Zm(self.omega[_k])] = cell.Zs_harm_fn(mi, omega_k)
            for mi, Ym in enumerate(Yg_m, start=1):
                subs[Ym(self.omega[_k])] = cell.Yg_harm_fn(mi, omega_k)

        if id(T_sym) not in self._lambdify_cache:
            self._build_evaluator(T_sym)
        fn, keys = self._lambdify_cache[id(T_sym)]
        return np.array(fn(*[subs[_k] for _k in keys]), dtype=complex)

    def build_cell_freq_matrices(
        self,
        T_sym: Matrix,
        dim: int,
        M: int,
        ks_state: list[int],
        Zs_m: list[type[Function]],
        Yg_m: list[type[Function]],
        w_s: np.ndarray,
        w_p: float,
        cells: list[CellImmitance],
    ) -> np.ndarray:
        """
        Evaluate T_sym on a (Nf x Nc) grid, vectorized over frequencies.

        For each cell the lambdified matrix is evaluated with all Nf frequency
        values at once (numpy broadcasting), reducing Python call overhead from
        Nf*Nc to Nc.

        Returns
        -------
        T_grid : np.ndarray, shape (Nf, Nc, dim, dim), dtype complex
        """
        w_s = np.asarray(w_s)
        Nf, Nc = len(w_s), len(cells)
        T_grid = np.empty((Nf, Nc, dim, dim), dtype=complex)

        N_max = max(abs(k) for k in ks_state) if ks_state else 0
        k_range = range(-(N_max + 2 * M), N_max + 2 * M + 1)

        if id(T_sym) not in self._lambdify_cache:
            self._build_evaluator(T_sym)
        fn, keys = self._lambdify_cache[id(T_sym)]

        zeros = np.zeros(Nf, dtype=complex)
        for c, cell in enumerate(cells):
            subs = {
                self.theta: cell.theta,
                self.omega_p: w_p,
            }
            for k in k_range:
                omega_k = w_s + k * w_p
                subs[self.omega[k]] = omega_k
                subs[self.Zs0(self.omega[k])] = cell.Zs0_fn(omega_k)
                subs[self.Yg0(self.omega[k])] = cell.Yg0_fn(omega_k)
                for mi, Zm in enumerate(Zs_m, start=1):
                    subs[Zm(self.omega[k])] = cell.Zs_harm_fn(mi, omega_k)
                for mi, Ym in enumerate(Yg_m, start=1):
                    subs[Ym(self.omega[k])] = cell.Yg_harm_fn(mi, omega_k)

            # fn returns dim×dim nested list; entries may be scalar for constant
            # matrix elements, add zeros to broadcast all entries to shape (Nf,)
            raw = fn(*[subs[k] for k in keys])
            T_grid[:, c] = np.array(
                [[e + zeros for e in row] for row in raw], dtype=complex
            ).transpose(2, 0, 1)

        return T_grid

    def export_matrix_graphic(
        self,
        T_sym: Matrix,
        state_syms: list[Expr] | None = None,
        filename: str = "transfer_matrix.pdf",
        title: str = "Transfer Matrix",
        fontsize: int = 9,
    ) -> Path:
        """
        Render T_sym as a PDF figure.

        Parameters
        ----------
        T_sym       : sympy Matrix
        state_syms  : list of state symbols used as row/column labels (optional)
        filename    : output path — .pdf, .png, or .svg
        title       : figure title
        fontsize    : base font size for cell content (pt)

        Returns
        -------
        pathlib.Path of the saved figure
        """
        import re
        import matplotlib

        matplotlib.rcParams["text.usetex"] = False
        matplotlib.rcParams["mathtext.fontset"] = "stix"
        matplotlib.rcParams["font.family"] = "STIXGeneral"
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from sympy import latex as sym_latex

        def _mathtext(expr):
            """sympy LaTeX → matplotlib-mathtext compatible string."""
            tex = sym_latex(expr)
            # \operatorname{X} is not supported by mathtext → \mathrm{X}
            tex = re.sub(r"\\operatorname\{([^}]+)\}", r"\\mathrm{\1}", tex)
            return tex

        dim = T_sym.shape[0]
        cells = [[_mathtext(T_sym[i, j]) for j in range(dim)] for i in range(dim)]
        labels = (
            [_mathtext(s) for s in state_syms]
            if state_syms
            else [f"V_{{{j},n}}" for j in range(dim // 2)]
            + [f"I_{{{j},n}}" for j in range(dim // 2)]
        )

        # Figure geometry
        cell_w = 3.8  # width per column
        cell_h = 0.95  # height per row
        lpad = 1.8  # left margin (row labels)
        bpad = 0.65  # bottom margin (column labels)
        bkt = 0.30  # bracket margin on each side
        tpad = 0.55  # top margin (title)

        fig_w = lpad + bkt + dim * cell_w + bkt
        fig_h = tpad + dim * cell_h + bpad

        fig = plt.figure(figsize=(fig_w, fig_h))

        # Axes covering the matrix cell grid
        l = (lpad + bkt) / fig_w
        r = 1.0 - bkt / fig_w
        b = bpad / fig_h
        t = 1.0 - tpad / fig_h
        ax = fig.add_axes((l, b, r - l, t - b))
        ax.set_xlim(0, dim)
        ax.set_ylim(0, dim)
        ax.axis("off")

        for i in range(dim):
            for j in range(dim):
                ax.text(
                    j + 0.5,
                    dim - i - 0.5,
                    f"${cells[i][j]}$",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                )

        for j, lbl in enumerate(labels):
            ax.text(
                j + 0.5,
                -0.4,
                f"${lbl}$",
                ha="center",
                va="top",
                fontsize=fontsize - 1,
                color="#555555",
            )

        for i, lbl in enumerate(labels):
            ax.text(
                -0.3,
                dim - i - 0.5,
                f"${lbl}$",
                ha="right",
                va="center",
                fontsize=fontsize - 1,
                color="#555555",
            )

        for idx in range(1, dim):
            ax.axhline(idx, color="#cccccc", lw=0.5, zorder=0)
            ax.axvline(idx, color="#cccccc", lw=0.5, zorder=0)

        def _data_to_fig(x_d, y_d):
            """Convert axes data coords to figure [0,1] coords."""
            return (l + (x_d / dim) * (r - l), b + (y_d / dim) * (t - b))

        x_l, _ = _data_to_fig(0, 0)
        x_r, _ = _data_to_fig(dim, 0)
        _, y_b = _data_to_fig(0, 0)
        _, y_t = _data_to_fig(0, dim)

        gap = 0.010  # gap between cell edge and bracket
        serif = 0.020  # horizontal serif length
        lw = 2.0

        def _draw_bracket(x_stem, outward):
            """Draw one [ or ] bracket in figure coordinates."""
            kw = dict(
                transform=fig.transFigure,
                color="black",
                lw=lw,
                solid_capstyle="butt",
                clip_on=False,
            )
            fig.add_artist(Line2D([x_stem, x_stem], [y_b, y_t], **kw))
            fig.add_artist(Line2D([x_stem, x_stem + outward * serif], [y_t, y_t], **kw))
            fig.add_artist(Line2D([x_stem, x_stem + outward * serif], [y_b, y_b], **kw))

        _draw_bracket(x_l - gap, 1)  # left  [
        _draw_bracket(x_r + gap, -1)  # right ]

        fig.text(
            0.5,
            1.0 - 0.18 / fig_h,
            title,
            ha="center",
            va="top",
            fontsize=fontsize + 3,
            fontweight="bold",
        )

        out = Path(filename)
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Matrix plot saved to {out}")
        return out
