import numpy as np
from numpy import random as rnd
from scipy.optimize import linear_sum_assignment, minimize_scalar
from scipy.linalg import eig, svd
from numpy.lib.stride_tricks import sliding_window_view

def generate_synthetic_decay(k, tmax, delta_t, rng, return_parameters=False):
    """
    Generates a pseudo-random exponentially decaying resonance.
    """

    amp = rng.random()

    linewidth = rng.uniform(0.5, 2.0)
    decay = np.pi * linewidth

    phase = 0

    spectral_width = 1 / delta_t
    freq = rng.uniform(-spectral_width / 4, spectral_width / 4)

    d_k = amp * np.exp(1j * phase)
    z_k = np.exp((-decay + 1j * 2 * np.pi * freq) * delta_t)

    n = np.arange(tmax)
    fid = d_k * z_k**n

    if return_parameters:
        return fid, z_k, d_k

    return fid
          

def generate_synthetic_signal(order, noise_level=0, tmax=32768, delta_t=1/12000, seed=18, return_parameters=False, p0=0, p1=0):
    """
    Generates a synthetic FID containing multiple resonances.
    """

    rng = np.random.default_rng(seed)
    signal = np.zeros(tmax, dtype=complex)

    z_values = []
    d_values = []

    n = np.arange(tmax)

    for element in range(order):

        fid, z_k, d_k = generate_synthetic_decay(element, tmax, delta_t, rng, return_parameters=True)
        frequency = np.angle(z_k) / (2 * np.pi)

        phase = p0 + p1 * frequency
        d_k *= np.exp(1j * phase)

        signal += d_k * z_k**n

        z_values.append(z_k)
        d_values.append(d_k)

    signal_rms = np.sqrt(np.mean(np.abs(signal)**2))
    sigma = noise_level * signal_rms / np.sqrt(2)

    real_noise = sigma * rng.normal(size=tmax)
    imag_noise = 1j * sigma * rng.normal(size=tmax)

    signal += real_noise + imag_noise

    if return_parameters:
        return signal, np.array(z_values), np.array(d_values)

    return signal

def matrix_pencil(signal, order, n_datapoints=200, return_singular_values=False):

    signal = np.asarray(signal, dtype=complex)[:n_datapoints]
    M = len(signal)

    if M < 2 * order:
        raise ValueError("Not enough samples for the requested order.")

    L = M // 2

    windows = sliding_window_view(signal, L)

    H_0 = windows[:-1]
    H_1 = windows[1:]

    U, S, Vh = svd(H_0, full_matrices=False)

    U_N = U[:, :order]
    S_N = S[:order]
    V_N = Vh[:order, :].conj().T

    B = U_N.conj().T @ H_1 @ V_N

    z_k = eig(B, np.diag(S_N))[0]
    z_k = z_k / np.maximum(1, np.abs(z_k)) # Prevents exponential graph if z > 1
    n = np.arange(M)

    V = z_k[None, :] ** n[:, None]
    d_k = np.linalg.lstsq(V, signal, rcond=None)[0]

    if return_singular_values:
        return z_k, d_k, S

    else:
        return z_k, d_k

def retrieve_phases(phased_signal, order, n_datapoints = 200, method='circular', bounds=(0*np.pi, 2*np.pi), 
                    filter_decay=False, threshold=6, water_suppression=False, ppm_axis=None, water_region=(4.5, 5.1)):
    """
    This takes a phased signal, and returns our matrix pencil estimation of the zeroth and first order phase
    """

    z_k, d_k = matrix_pencil(phased_signal, order = order, n_datapoints = n_datapoints)

    if filter_decay:

        decay = -np.log(np.clip(np.abs(z_k), 1e-12, 1))
    
        median = np.median(decay)
        mad = np.median(np.abs(decay - median))
    
        if mad > 0:
            
            cutoff = median + threshold * 1.4826 * mad
            mask = decay <= cutoff
    
            if np.sum(mask) >= 3:
                z_k = z_k[mask]
                d_k = d_k[mask]

    if water_suppression:

        if ppm_axis is None:
            raise ValueError("ppm_axis is required for water suppression.")

        fft_axis = np.fft.fftshift(np.fft.fftfreq(len(ppm_axis)))

        pole_ppm = np.interp(np.angle(z_k) / (2*np.pi),fft_axis,np.asarray(ppm_axis)[::-1])
        mask = ((pole_ppm < water_region[0]) |(pole_ppm > water_region[1]))

        if np.sum(mask) < 2:
            raise ValueError("Insufficient components after water exclusion.")

        z_k = z_k[mask]
        d_k = d_k[mask]

    frequencies = np.angle(z_k) / (2 * np.pi)
    phases = np.angle(d_k)

    if method == 'linear':
    
        idx = np.argsort(frequencies)
        weights = np.abs(d_k)[idx]
    
        x = frequencies[idx]
        y = np.unwrap(phases[idx])
          
        recovered_phi_1, recovered_phi_0 = np.polyfit(x, y, 1, w=weights)
        recovered_phi_0 = np.angle(np.exp(1j * recovered_phi_0))
    
        return recovered_phi_0, recovered_phi_1

    if method == 'circular':
        
        weights = np.abs(d_k)**2

        def objective(p1):
            return -np.abs(np.sum(weights * np.exp(1j * (phases - p1 * frequencies))))
    
        grid = np.linspace(*bounds, 1001)
        losses = np.array([objective(p1) for p1 in grid])
    
        best = np.argmin(losses)
    
        left = grid[max(best - 1, 0)]
        right = grid[min(best + 1, len(grid) - 1)]
    
        result = minimize_scalar(objective, bounds=(left, right), method="bounded", options={"xatol": 1e-12})
    
        candidates = [bounds[0], bounds[1], result.x]
        p1_est = min(candidates, key=objective)
    
        p0_est = np.angle(np.sum(weights * np.exp(1j * (phases - p1_est * frequencies))))
    
        return p0_est, p1_est