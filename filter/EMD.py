import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.signal import find_peaks, savgol_filter
import sys
sys.path.append(r"../src/Filters")
from MDN_bandstop import normalize_to_peak, compute_welch_spectrum

def find_extrema(x):
    maxima = np.where((x[1:-1] > x[0:-2]) & (x[1:-1] > x[2:]))[0] + 1
    minima = np.where((x[1:-1] < x[0:-2]) & (x[1:-1] < x[2:]))[0] + 1
    return maxima, minima

# Volgende drie functies nodig om na te gaan of potentiële IMFs ook echt IMFs zijn
def count_zero_crossings(x):
    signs = np.sign(x)
    sign_diff = np.diff(signs)
    crossings = sign_diff != 0
    num_crossings = np.sum(crossings)
    return num_crossings

def count_extrema(maxima, minima):
    return len(maxima) + len(minima)

def compute_potential_imf(x, time):
    maxima, minima = find_extrema(x)
    # max_line = interp1d(time[maxima], x[maxima], kind='cubic', fill_value="extrapolate") 
    # min_line = interp1d(time[minima], x[minima], kind='cubic', fill_value="extrapolate")
    max_line = PchipInterpolator(time[maxima], x[maxima], extrapolate=True)
    min_line = PchipInterpolator(time[minima], x[minima], extrapolate=True)
    avg_line = 0.5 * (max_line(time) + min_line(time))
    # avg_line = (max_line(time) + min_line(time)) / 2
    
    potential_IMF = x - avg_line
    
    num_zero_crossings = count_zero_crossings(potential_IMF)
    sum_max_min = count_extrema(maxima, minima)
    avg_value = np.mean(potential_IMF)
    extrema_ok = abs(sum_max_min - num_zero_crossings) <= 1
    mean_ok = np.isclose(avg_value, 0, atol=1e-10)
    
    return potential_IMF, extrema_ok, mean_ok

# Dit hieronder zijn vervolgstappen als potentiële IMF ook IMF blijkt; stap 5a en 5b & 6

def emd(signal, time, max_imfs=10, max_siftings=100, stop_std=0.1):
    """
    Perform Empirical Mode Decomposition (EMD) on a 1D signal.
    
    Parameters:
        signal:  input signal x(t)
        time: corresponding time vector
        max_imfs: maximum number of IMFs to extract
        max_siftings: max iterations per IMF (sifting)
        stop_std: threshold for stopping sifting (residual energy)
    
    Returns:
        imfs: list of extracted IMFs
        residual: final residual signal
    """
    residual = signal.copy()
    imfs = []

    for n in range(max_imfs):
        h = residual.copy()
        for s in range(max_siftings):
            potential_h, extrema_ok, mean_ok = compute_potential_imf(h, time)
            h = potential_h
            if extrema_ok and mean_ok:
                break  # IMF found
        
        # Storing IMF - stap 5a
        imfs.append(h)
        residual = residual - h

        # Check stopping criterion (residual energy small)
        if np.allclose(residual, 0, atol=stop_std):
            break

    return imfs, residual


def reconstruct_signal(signal, imfs, num_imfs_to_subtract=3):
    """
    Reconstrueer een signaal door de eerste num_imfs_to_subtract IMFs af te trekken.

    Parameters:
        signal : np.ndarray
            Origineel signaal
        imfs : list of np.ndarray
            Lijst van geëxtraheerde IMFs
        num_imfs_to_subtract : int
            Aantal eerste IMFs dat wordt afgetrokken

    Returns:
        reconstructed_signal : np.ndarray
            Het gereconstrueerde (gefilterde) signaal
    """
    num_imfs = len(imfs)
    num_to_subtract = min(num_imfs_to_subtract, num_imfs)

    if num_to_subtract > 0:
        reconstructed_signal = signal - np.sum(imfs[:num_to_subtract], axis=0)
    else:
        reconstructed_signal = signal.copy()

    return reconstructed_signal


def estimate_fundamental_frequency(
    pes,time,
    fs=256,
    max_imfs=3,
    fmin=0.67,
    fmax=4.0,
    min_prom_factor=0.1):
    """
    Schat fundamentele hartfrequentie uit PES m.b.v. EMD + PSD.

    Parameters
    ----------
    pes : np.ndarray
        Oesophageal pressure signal
    fs : float
        Sampling frequency [Hz]

    Returns
    -------
    fundamental_freq : float or None
        Fundamentele frequentie in Hz (None indien niet gevonden)
    """

    # -----------------------------
    # 1) EMD
    # -----------------------------
    imfs, _ = emd(
        pes,
        time,
        max_imfs=max_imfs,
        max_siftings=200,
        stop_std=0.05
    )

    if len(imfs) < 3:
        return None

    # -----------------------------
    # 2) PSD helper functies
    # -----------------------------
    def smooth_psd(mag):
        N = len(mag)
        wl = int(0.3 * N)      # 3% van PSD lengte
        wl = max(5, min(wl, 3))
        if wl % 2 == 0:
            wl += 1
        return savgol_filter(mag, wl, 3)

    def detect_peaks(freq, mag, fmin=0.67, fmax=4.0, min_prom_factor=0.1):
        mask = (freq >= fmin) & (freq <= fmax)
        f = freq[mask]
        m = mag[mask]
        m_large = m #relatief vergroten van de pieken
        prom = min_prom_factor * (np.max(m_large) - np.min(m_large))
        peaks, props = find_peaks(m_large, prominence=prom)

        return f, m, peaks, props

    # ---------------------------------------------------------
    # 3) PSD IMF1 + IMF2 (jouw eigen Welch functie)
    # ---------------------------------------------------------
    freq1, mag1 = compute_welch_spectrum(imfs[0], fs=256)
    freq2, mag2 = compute_welch_spectrum(imfs[1], fs=256)
    freq3, mag3 = compute_welch_spectrum(imfs[2], fs=256)
    mag1 = normalize_to_peak(mag1)
    mag2 = normalize_to_peak(mag2)
    mag3 = normalize_to_peak(mag3)

    # ---------------------------------------------------------
    # 4) Smoothen
    # ---------------------------------------------------------
    mag1_sm = smooth_psd(mag1)
    mag2_sm = smooth_psd(mag2)
    mag3_sm = smooth_psd(mag3)

    # ---------------------------------------------------------
    # 5) Pieken detecteren
    # ---------------------------------------------------------
    f1, m1, peaks1, props1 = detect_peaks(freq1, mag1_sm)
    f2, m2, peaks2, props2 = detect_peaks(freq2, mag2_sm)
    f3, m3, peaks3, props3 = detect_peaks(freq3, mag3_sm)

    mag4_sm = mag1_sm + mag2_sm - mag3_sm
    f4, m4, peaks4, props4 = detect_peaks(freq1, mag4_sm)
    # ---------------------------------------------------------
    # 6) Fundamentele frequentie bepalen
    # ---------------------------------------------------------
    if len(peaks1) == 0:
        print("⚠️ Geen pieken gevonden in IMF1.")
        fundamental_freq = None
    else:
        fundamental_freq = f4[peaks4[0]]
        print("Fundamentele hartfrequentie:", fundamental_freq, "Hz")
        print("≈", fundamental_freq * 60, "BPM")

        # check IMF2 bevestiging
        if len(peaks2) > 0:
            imf2_freqs = f2[peaks2]
            if np.any(np.abs(imf2_freqs - fundamental_freq) < 0.1):
                print("→ Bevestigd door IMF2 (peak at same frequency)")
            else:
                print("→ Niet bevestigd door IMF2")

    return fundamental_freq

