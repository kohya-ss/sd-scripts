import numpy as np

from library.timestep_visualization import timesteps_to_bin_indices


def test_near_integer_timesteps_are_rounded():
    # float32 grid values like 250.99998 (from sigma * 1000) must be counted at 251, not 250
    ts = np.array([250.99998, 251.0, 501.0, 500.99997, 0.4, 0.6, 999.6, 1000.0], dtype=np.float32)
    assert timesteps_to_bin_indices(ts, 1000).tolist() == [251, 251, 501, 501, 0, 1, 999, 999]


def test_grid_from_float32_sigmas_has_no_double_bins():
    timesteps = (np.arange(1, 1001, dtype=np.float32) / np.float32(1000.0)) * np.float32(1000.0)
    counts = np.bincount(timesteps_to_bin_indices(timesteps, 1000), minlength=1000)
    # every bin 1..998 gets exactly one grid point; 999 gets both 999 and 1000 (clipped), 0 gets nothing
    assert counts[1:999].tolist() == [1] * 998
    assert counts[999] == 2 and counts[0] == 0
