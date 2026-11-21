import numpy as np
from mimo_system import (
    CLASS_TO_SYMBOLS, ls_channel_estimate, ls_channel_estimate_orthogonal,
    pilot_matrix, simulate_block, symbols_to_labels,
)
def main():
    rng = np.random.default_rng(12345)
    xp = pilot_matrix(8)
    assert xp.shape == (2,8)
    assert np.allclose(xp @ xp.T, 8*np.eye(2))
    b = simulate_block(20, rng, 8, 256)
    assert b["features"].shape == (256,12)
    assert b["h"].shape == b["h_hat"].shape == (2,2)
    assert np.allclose(
        ls_channel_estimate(b["yp"], b["xp"]),
        ls_channel_estimate_orthogonal(b["yp"], b["xp"]),
        atol=1e-12
    )
    assert np.array_equal(symbols_to_labels(CLASS_TO_SYMBOLS), np.arange(4))
    print("Xp Xp^H =\n", xp @ xp.T)
    print("NMSE at 20 dB =", b["nmse"])
    print("[PASS] Core MIMO + pilots + LS + 12-feature mapping is consistent.")
if __name__ == "__main__":
    main()
