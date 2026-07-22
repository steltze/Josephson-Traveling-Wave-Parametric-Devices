import numpy as np

def single_mode_matrix(Zs, Yg):
    # Zs, Yg: (Nf, m, m)
    Nf, m, _ = Zs.shape
    I = np.broadcast_to(np.eye(m, dtype=complex), (Nf, m, m))
    top = np.concatenate([I, -Zs], axis=-1)
    bottom = np.concatenate([-Yg, I + Yg @ Zs], axis=-1)
    return np.concatenate([top, bottom], axis=-2) # (Nf, n, n)

def symmetric_single_mode_matrix(Zs, Yg):
    # Zs, Yg: (Nf, m, m)
    # Symmetric pi-cell: shunt Yg/2 - series Zs - shunt Yg/2
    Nf, m, _ = Zs.shape
    I = np.broadcast_to(np.eye(m, dtype=complex), (Nf, m, m))
    Yg_half = Yg / 2
    top = np.concatenate([I + Zs @ Yg_half, -Zs], axis=-1)
    bottom = np.concatenate([-Yg - Yg_half @ Zs @ Yg_half, I + Yg_half @ Zs], axis=-1)
    return np.concatenate([top, bottom], axis=-2) # (Nf, n, n)

def single_mode_matrix_grid(cells, topology):

    Ncells = len(cells)
    Nf, m, _ = cells[0].Zs_harm_fn.shape
    T_grid = np.empty((Nf, Ncells, 2 * m, 2 * m), dtype=complex)

    if topology == "L":
        for c, cell in enumerate(cells):
            T_grid[:, c] = single_mode_matrix(cell.Zs_harm_fn, cell.Yg_harm_fn) # (Nf, n, n)
    else:
        for c, cell in enumerate(cells):
            T_grid[:, c] = symmetric_single_mode_matrix(cell.Zs_harm_fn, cell.Yg_harm_fn) # (Nf, n, n)
    return T_grid # (Nf, Ncells, n, n)
