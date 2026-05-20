"""
Symbolic transfer-matrix builder for TWPA/TWPC unit cells for time-periodic impedance/admittance networks.
The transfer matrix T maps the state vector [V[k,n], I[k,n]] -> [V[k,n+1], I[k,n+1]] for each sideband k simultaneously.
"""

from sympy import symbols, IndexedBase, Function, exp, I, Add, expand, zeros
import numpy as np


# Shared symbolic variables

t, theta, omega_p = symbols('t theta omega_p', real=True)
k, n              = symbols('k n', integer=True)
omega             = IndexedBase('omega')
Zs0               = symbols('Zs0', complex=True)
Yg0               = symbols('Yg0', complex=True)
V                 = IndexedBase('V')
Ic                = IndexedBase('I')
xi                = symbols('xi')   # placeholder frequency argument inside series


def fourier_basis(m):
    """Return the m-th basis element exp(I*m*omega_p*t)."""
    return exp(I * m * omega_p * t)


def make_harmonic_functions(M):
    """Return lists of SymPy Function objects Zs_m and Yg_m for orders 1..M."""
    Zs_m = [Function(f"Zs{m}") for m in range(1, M + 1)]
    Yg_m = [Function(f"Yg{m}") for m in range(1, M + 1)]
    return Zs_m, Yg_m


def build_Zs_series(Zs_m):
    """
    Return the symbolic Zs(t) Fourier series up to order M.

    Uses the module-level symbols t, theta, omega_p, Zs0, xi.
    The placeholder xi is later substituted with omega[k] when extracting
    harmonic coefficients.
    """
    impedance = Zs0
    for mi, Zm in enumerate(Zs_m, start=1):
        impedance += Zm(xi) * fourier_basis(mi) * exp(I * mi * theta) + Zm(
            xi
        ) * fourier_basis(-mi) * exp(-I * mi * theta)
    return impedance


def build_Yg_shunt(Yg_m):
    """
    Return the symbolic Yg(t) Fourier series up to order M.

    Uses the module-level symbols t, theta, omega_p, Yg0, xi.
    """
    admittance = Yg0
    for mi, Ym in enumerate(Yg_m, start=1):
        admittance += Ym(xi) * fourier_basis(mi) * exp(I * mi * theta) + Ym(
            xi
        ) * fourier_basis(-mi) * exp(-I * mi * theta)
    return admittance


def extract_harmonic_results(M, Zs_t, Yg_t):
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
    Vn_1 = V[k, n] - Zs_t * Ic[k, n]
    In_1 = Ic[k, n] - Yg_t * Vn_1

    Vn_1_expanded = expand(Vn_1)
    In_1_expanded = expand(In_1)

    ms_nonzero = [m for m in range(-2 * M, 2 * M + 1) if m != 0]
    results = {}

    for m in ms_nonzero:
        cv_raw = Vn_1_expanded.coeff(fourier_basis(m))
        ci_raw = In_1_expanded.coeff(fourier_basis(m))
        cv = cv_raw.subs(xi, omega[k]).subs(k, k - m)
        ci = ci_raw.subs(xi, omega[k]).subs(k, k - m)
        results[m] = (cv, ci)

    # DC component: subtract all oscillating parts
    V_osc = Add(
        *[Vn_1_expanded.coeff(fourier_basis(m)) * fourier_basis(m) for m in ms_nonzero]
    )
    I_osc = Add(
        *[In_1_expanded.coeff(fourier_basis(m)) * fourier_basis(m) for m in ms_nonzero]
    )
    cv0 = expand(Vn_1_expanded - V_osc).subs(xi, omega[k])
    ci0 = expand(In_1_expanded - I_osc).subs(xi, omega[k])
    results[0] = (cv0, ci0)

    return results


def build_transfer_matrix(M, ks_state, results):
    """
    Build the symbolic transfer matrix T.

    Parameters
    ----------
    M        : truncation order (determines harmonic range ±2M)
    ks_state : list of sideband indices; each contributes a (V[k,n], I[k,n]) pair
    results  : dict from extract_harmonic_results

    Returns
    -------
    T_sym      : sympy Matrix of shape (2*len(ks_state), 2*len(ks_state))
    state_syms : list of state symbols in row/column order
    """
    state_syms = []
    for _k in ks_state:
        state_syms += [V[_k, n], Ic[_k, n]]
    dim = len(state_syms)

    ms_full = list(range(-2 * M, 2 * M + 1))
    T_sym = zeros(dim, dim)

    for row_idx, _k in enumerate(ks_state):
        cv_p = expand(Add(*[results[m][0].subs(k, _k) for m in ms_full]))
        ci_p = expand(Add(*[results[m][1].subs(k, _k) for m in ms_full]))
        for col_idx, s in enumerate(state_syms):
            T_sym[2 * row_idx, col_idx] = cv_p.coeff(s)
            T_sym[2 * row_idx + 1, col_idx] = ci_p.coeff(s)

    return T_sym, state_syms


def build_symbolic_transfer_matrix(M, ks_state):
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
    Zs_m, Yg_m = make_harmonic_functions(M)
    Zs_t = build_Zs_series(Zs_m)
    Yg_t = build_Yg_shunt(Yg_m)
    results = extract_harmonic_results(M, Zs_t, Yg_t)
    T_sym, state_syms = build_transfer_matrix(M, ks_state, results)
    return T_sym, state_syms, Zs_m, Yg_m


def build_numeric_matrix(
    T_sym,
    dim,
    M,
    ks_state,
    Zs_m,
    Yg_m,
    Zs0_val,
    Yg0_val,
    theta_val,
    omega_p_val,
    omega_val_fn,
    Zs_num_fn,
    Yg_num_fn,
    k_val=0,
):
    """
    Substitute numerical values into T_sym and return a numpy complex array.

    Parameters
    ----------
    T_sym        : sympy Matrix from build_symbolic_transfer_matrix
    dim          : int, matrix size (= 2*len(ks_state))
    M            : Floquet truncation order
    ks_state     : list of sideband indices
    Zs_m, Yg_m  : harmonic Function objects from make_harmonic_functions
    Zs0_val      : complex, DC series impedance
    Yg0_val      : complex, DC shunt admittance
    theta_val    : float, pump phase
    omega_p_val  : float, pump angular frequency
    omega_val_fn : callable  q -> float  (carrier frequency of mode q)
    Zs_num_fn    : callable  (m, freq) -> complex  (m-th Zs Fourier coefficient)
    Yg_num_fn    : callable  (m, freq) -> complex  (m-th Yg Fourier coefficient)
    k_val        : int, center mode index

    Returns
    -------
    T_num : numpy ndarray, shape (dim, dim), dtype complex
    """
    N_max = max(abs(_k) for _k in ks_state) if ks_state else 0
    q_range = range(k_val - (N_max + 2 * M), k_val + (N_max + 2 * M) + 1)

    subs = {
        Zs0: Zs0_val,
        Yg0: Yg0_val,
        theta: theta_val,
        omega_p: omega_p_val,
        k: k_val,
    }
    for q in q_range:
        subs[omega[q]] = omega_val_fn(q)
        freq = omega_val_fn(q)
        for mi, Zm in enumerate(Zs_m, start=1):
            subs[Zm(omega[q])] = Zs_num_fn(mi, freq)
        for mi, Ym in enumerate(Yg_m, start=1):
            subs[Ym(omega[q])] = Yg_num_fn(mi, freq)

    T_num = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            T_num[i, j] = complex(T_sym[i, j].subs(subs))
    return T_num


def export_matrix_plot(
    T_sym,
    state_syms=None,
    filename="transfer_matrix.pdf",
    title="Transfer Matrix",
    fontsize=9,
):
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
    from pathlib import Path
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
        else [str(j) for j in range(dim)]
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
    ax = fig.add_axes([l, b, r - l, t - b])
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
