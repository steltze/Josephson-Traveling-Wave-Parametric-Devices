import numpy as np
import pytest

from simulation.config import SimulationConfig


class TestFrequencyGrid:
    def test_omegas_is_2pi_times_freqs(self):
        cfg = SimulationConfig(freq_min=1.0, freq_max=5.0, n_freqs=9)
        np.testing.assert_allclose(cfg.omegas, cfg.freqs * 2 * np.pi)

    def test_freqs_spans_requested_range(self):
        cfg = SimulationConfig(freq_min=2.0, freq_max=8.0, n_freqs=4)
        np.testing.assert_allclose(cfg.freqs, [2.0, 4.0, 6.0, 8.0])


class TestVelocities:
    def test_v_signal_formula(self):
        cfg = SimulationConfig(cell_size=1e-5, omega_cutoff=100.0)
        assert cfg.v_signal == pytest.approx(1e-5 * 100.0 / 2)

    def test_v_pump_single_tone(self):
        cfg = SimulationConfig(cell_size=1e-5, omega_cutoff=100.0, v_ratio=4.0)
        assert cfg.v_pump == pytest.approx(cfg.v_signal / 4.0)

    def test_v_pump_multi_pump_matches_per_pump_ratio(self):
        cfg = SimulationConfig(
            epsilon=[0.1, 0.05],
            v_ratio=[2.0, 5.0],
            omega_pump=[10.0, 20.0],
        )
        v_pump = cfg.v_pump
        assert v_pump == pytest.approx([cfg.v_signal / 2.0, cfg.v_signal / 5.0])

    def test_propagation_direction_co_propagating(self):
        cfg = SimulationConfig(v_ratio=3.0)
        assert cfg.propagation_direction == 1.0

    def test_propagation_direction_counter_propagating(self):
        cfg = SimulationConfig(v_ratio=-3.0)
        assert cfg.propagation_direction == -1.0

    def test_propagation_direction_uses_first_pump_in_multi_pump(self):
        cfg = SimulationConfig(
            epsilon=[0.1, 0.05], v_ratio=[-2.0, 5.0], omega_pump=[10.0, 20.0]
        )
        assert cfg.propagation_direction == -1.0


class TestMultiPumpValidation:
    def test_v_ratio_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="v_ratio"):
            SimulationConfig(epsilon=[0.1, 0.2], v_ratio=[1.0], omega_pump=[1.0, 2.0])

    def test_omega_pump_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="omega_pump"):
            SimulationConfig(
                epsilon=[0.1, 0.2], v_ratio=[1.0, 2.0], omega_pump=[1.0]
            )

    def test_kmax_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Kmax"):
            SimulationConfig(
                epsilon=[0.1, 0.2],
                v_ratio=[1.0, 2.0],
                omega_pump=[1.0, 2.0],
                Kmax=[(-1, 1)],
            )

    def test_kmax_inverted_range_raises(self):
        with pytest.raises(ValueError, match="k_min <= k_max"):
            SimulationConfig(
                epsilon=[0.1, 0.2],
                v_ratio=[1.0, 2.0],
                omega_pump=[1.0, 2.0],
                Kmax=[(2, -2), (0, 1)],
            )

    def test_valid_multi_pump_config_does_not_raise(self):
        SimulationConfig(
            epsilon=[0.1, 0.2],
            v_ratio=[1.0, 2.0],
            omega_pump=[1.0, 2.0],
            Kmax=[(-1, 1), (0, 1)],
        )
