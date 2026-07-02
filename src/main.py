import numpy as np
import matplotlib.pyplot as plt

from logger import get_logger, setup_logging

log = get_logger(__name__)
from simulation import SimulationConfig, Simulation
from models import JTLDiscrete
from symbolic import CellSingleMode
from examples.julia_comparison import julia_comparison
from examples.harmonics_comparison import harmonics_comparison
from examples.track_gap_center_over_pump_frequency import (
    track_gap_center_over_pump_frequency,
)
from examples.discrete_vs_continuous import continuous_vs_discrete
from examples.compare_gap_bandwidth import compare_gap_bandwidth
from examples.compare_dispersion_relations import compare_dispersion_relations
from examples.utils import save_all


def main():
    cfg = SimulationConfig(
        Z0=50,
        M=1,
        ks_state=[0, 1],
        ncell=1000,
        cell_size=10e-6,
        omega_cutoff=50 * 2 * np.pi,  # RH ATL
        # omega_cutoff=0.1 * 2 * np.pi, # LH ATL
        omega_pump=6.8 * 2 * np.pi,
        omega_j=60 * 2 * np.pi,
        epsilon=0.0,
        omega_c=5 * 2 * np.pi,
        v_ratio=2.5,
        freq_min=1,
        freq_max=12,
        n_freqs=500,
        disorder=False,
        nramp=0.0,
    )

    log.info("omega_pump = %.3f GHz", cfg.omega_pump / (2 * np.pi * 1e9))

    # sim = Simulation(JTLDiscrete.left_handed(), cfg)
    sim = Simulation(JTLDiscrete, cfg)

    symbolic_matrix, _ = sim.get_symbolic_matrix()

    symbolic_matrix, state_syms = sim.get_symbolic_matrix()
    # T_inv = symbolic_matrix.inv()
    import sympy
    sympy.latex(symbolic_matrix)
    a = CellSingleMode()
    a.export_matrix_graphic(symbolic_matrix, state_syms)

    sim.plot_s_parameters([(1, 1), (3, 1)])
    sim.plot_s_parameters([(2, 2), (4, 2)])
    sim.plot_dispersion_relation()
    # save_all("main")
    plt.show()


if __name__ == "__main__":
    setup_logging()

    # main()

    julia_comparison()

    # harmonics_comparison()

    # track_gap_center_over_pump_frequency()

    # continuous_vs_discrete()

    # compare_gap_bandwidth()

    # compare_dispersion_relations()

# linear in epsilon, kerr (no cross modulation, 3 wave mixing)
# superimpose with center bandwidth
# photon number conservation, start with pump near zero, no modulation etc
# normalize the transfer matrices for photon flux
