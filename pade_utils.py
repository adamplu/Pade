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

def pade_solve(signal, order, datapoints, tmax=1000, delta_t=0.01, svd=False):
    """
    Given a signal and the number of order to decompose into, this uses the Pade techniques detailed in the previous notebook 
    to reconstruct (and return) the original signals. 
    """

    signal_used = signal[:datapoints].tolist()
    empty_matrix = []
    rhs_vector = []

    for i in range(datapoints - order):
        matrix_row = signal_used[i:order+i]
        matrix_row.reverse()
        empty_matrix.append(matrix_row)
        rhs_vector.append( -1 * signal_used[order+i])

    matrix = np.array(empty_matrix, dtype=complex)
    rhs_vector = np.array(rhs_vector, dtype=complex)

    scales = np.linalg.norm(matrix, axis = 0)
    matrix_scaled = matrix / scales

    if svd == True:

        U, s, Vh = np.linalg.svd(matrix_scaled, full_matrices=False)
        threshold = 1e-15 * s[0]
        s_inv = np.array([1 / value if value > threshold else 0 for value in s])
        q_k_scaled = Vh.conj().T @ np.diag(s_inv) @ U.conj().T @ rhs_vector
    
    else:

        q_k_scaled, _, _, _ = np.linalg.lstsq(matrix_scaled, rhs_vector, rcond=None)

    q_k = q_k_scaled / scales

    coefficients = np.append(q_k[::-1], 1)
    roots = np.roots(coefficients)

    z_k = [(1 / root) for root in roots]

    empty_matrix = []
    rhs_vector = []

    for i in range(datapoints):
        matrix_row = [z**i for z in z_k]
        empty_matrix.append(matrix_row)
        rhs_vector.append(signal[i])

    matrix = np.array(empty_matrix, dtype=complex)
    rhs_vector = np.array(rhs_vector, dtype=complex)

    scales = np.linalg.norm(matrix, axis = 0)
    matrix_scaled = matrix / scales

    if svd == True:

        U, s, Vh = np.linalg.svd(matrix_scaled, full_matrices=False)
        threshold = 1e-15 * s[0]
        s_inv = np.array([1 / value if value > threshold else 0 for value in s])
        d_k_scaled = Vh.conj().T @ np.diag(s_inv) @ U.conj().T @ rhs_vector
    
    else:

        d_k_scaled, _, _, _ = np.linalg.lstsq(matrix_scaled, rhs_vector, rcond=None)

    d_k = d_k_scaled / scales

    mag = np.abs(z_k)
    arg = np.angle(z_k, deg=False)

    Lambda = - (np.log(mag)) / delta_t
    Frequency = arg / (2*np.pi*delta_t)

    Amplitude = np.abs(d_k)
    Phase = np.angle(d_k, deg=False)

    assert len(z_k) == len(d_k)

    individual_signals = []

    for z, d in zip(z_k, d_k):

        signal = []

        for n in range(tmax):
            signal.append(d * z**n)

        individual_signals.append(signal)

    individual_signals = np.array(individual_signals, dtype=complex)

    return individual_signals

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