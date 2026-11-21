import numpy as np
from config import NT, NR, NP_DEFAULT, ND_PER_BLOCK

CLASS_TO_SYMBOLS = np.array(
    [[-1.0, -1.0], [-1.0, +1.0], [+1.0, -1.0], [+1.0, +1.0]],
    dtype=np.float64,
)

def pilot_matrix(npilots=NP_DEFAULT):
    if npilots % 2:
        raise ValueError("Np must be even for this orthogonal BPSK construction.")
    xp = np.vstack([
        np.ones(npilots, dtype=np.float64),
        np.where(np.arange(npilots) % 2 == 0, 1.0, -1.0),
    ])
    if not np.allclose(xp @ xp.T, npilots * np.eye(NT)):
        raise RuntimeError("Pilot orthogonality check failed.")
    return xp

def snr_db_to_noise_variance(snr_db):
    return 10.0 ** (-float(snr_db) / 10.0)

def complex_awgn(shape, sigma2, rng):
    s = np.sqrt(sigma2 / 2.0)
    return s * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

def rayleigh_channel(rng):
    s = 1.0 / np.sqrt(2.0 * NT)
    return s * (rng.standard_normal((NR, NT)) + 1j * rng.standard_normal((NR, NT)))

def transmit(h, x, sigma2, rng):
    return h @ x + complex_awgn((NR, x.shape[1]), sigma2, rng)

def ls_channel_estimate(yp, xp):
    gram = xp @ xp.conj().T
    return yp @ xp.conj().T @ np.linalg.inv(gram)

def ls_channel_estimate_orthogonal(yp, xp):
    return (yp @ xp.conj().T) / xp.shape[1]

def nmse_per_realization(h, h_hat):
    return float(np.linalg.norm(h-h_hat, "fro")**2 / np.linalg.norm(h, "fro")**2)

def labels_to_symbols(labels):
    return CLASS_TO_SYMBOLS[np.asarray(labels, dtype=np.int64)]

def symbols_to_labels(symbols):
    s = np.asarray(symbols)
    if s.ndim == 2 and s.shape[0] == 2:
        s = s.T
    b0 = (s[:, 0] > 0).astype(np.int64)
    b1 = (s[:, 1] > 0).astype(np.int64)
    return 2*b0 + b1

def random_bpsk_vectors(n, rng):
    labels = rng.integers(0, 4, size=n, dtype=np.int64)
    return labels_to_symbols(labels).T, labels

def make_features(y, h_hat):
    n = y.shape[1]
    hf = np.concatenate([np.real(h_hat).reshape(-1), np.imag(h_hat).reshape(-1)])
    f = np.empty((n, 12), dtype=np.float32)
    f[:, 0:2] = np.real(y).T
    f[:, 2:4] = np.imag(y).T
    f[:, 4:] = hf[None, :]
    return f

def simulate_pilot_estimation(snr_db, rng, npilots=NP_DEFAULT):
    sigma2 = snr_db_to_noise_variance(snr_db)
    h = rayleigh_channel(rng)
    xp = pilot_matrix(npilots)
    yp = transmit(h, xp, sigma2, rng)
    h_hat = ls_channel_estimate_orthogonal(yp, xp)
    return h, h_hat, xp, yp, sigma2

def simulate_block(snr_db, rng, npilots=NP_DEFAULT, ndata=ND_PER_BLOCK):
    h, h_hat, xp, yp, sigma2 = simulate_pilot_estimation(snr_db, rng, npilots)
    x, labels = random_bpsk_vectors(ndata, rng)
    y = transmit(h, x, sigma2, rng)
    return dict(
        h=h, h_hat=h_hat, xp=xp, yp=yp, x=x, y=y, labels=labels,
        features=make_features(y, h_hat), sigma2=sigma2,
        nmse=nmse_per_realization(h, h_hat),
    )

def detect_zf(y, h_hat):
    xt = np.linalg.pinv(h_hat) @ y
    return np.where(np.real(xt) >= 0, 1.0, -1.0)

def detect_mmse(y, h_hat, sigma2, es=1.0):
    a = h_hat.conj().T @ h_hat + (sigma2/es) * np.eye(NT)
    b = h_hat.conj().T @ y
    xt = np.linalg.solve(a, b)
    return np.where(np.real(xt) >= 0, 1.0, -1.0)
