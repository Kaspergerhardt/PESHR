import numpy as np
from scipy.signal import find_peaks

def find_peak_properties_bandstop_prev(freqs, mag, f_min=0.8, f_max=4.0, percent_of_peak=0.5):
    """
    Zoek de hoogste piek binnen een gegeven frequentieband en bereken fundamentele frequentie,
    amplitude en piekbreedte bij een bepaalde drempel (percent_of_peak).

    Parameters
    ----------
    freqs : array-like
        Frequencies van het spectrum.
    mag : array-like
        Magnitude van het spectrum.
    f_min : float
        Minimum frequentie voor zoekgebied (Hz).
    f_max : float
        Maximum frequentie voor zoekgebied (Hz).
    percent_of_peak : float, optional
        Percentage van piekamplitude om piekbreedte te definiëren (default=0.2).

    Returns
    -------
    fundamental_freq : float
        Frequentie van de hoogste piek binnen de band.
    fundamental_amp : float
        Amplitude van de piek.
    peak_width : float
        Breedte van de piek bij amplitude ≤ percent_of_peak * piekamplitude.
    left_freq : float
        Linkergrens van piek bij threshold.
    right_freq : float
        Rechtergrens van piek bij threshold.
    """
    # Beperk tot band
    mask = (freqs >= f_min) & (freqs <= f_max)
    freqs_band = freqs[mask]
    mag_band = mag[mask]

    if len(freqs_band) == 0:
        raise ValueError("Geen frequenties binnen het opgegeven bereik.")

    # Hoogste piek binnen band
    peak_idx = np.argmax(mag_band)
    fundamental_freq = freqs_band[peak_idx]
    fundamental_amp = mag_band[peak_idx]
    threshold = percent_of_peak * fundamental_amp

    # Breedte rond piek berekenen binnen band
    below_thresh = mag_band <= threshold

    # Links van de piek
    left_candidates = np.where(below_thresh[:peak_idx])[0]
    left_idx = left_candidates[-1] if len(left_candidates) > 0 else 0

    # Rechts van de piek
    right_candidates = np.where(below_thresh[peak_idx:])[0]
    right_idx = peak_idx + right_candidates[0] if len(right_candidates) > 0 else len(mag_band)-1

    left_freq = freqs_band[left_idx]
    right_freq = freqs_band[right_idx]
    peak_width = right_freq - left_freq

    return fundamental_freq, fundamental_amp, peak_width, left_freq, right_freq


def remove_fundamental_and_harmonics_fft_prev(signal, fundamental_freq, fs = 256, bw=0.01):
    """
    Verwijder fundamentele frequentie en alle hogere harmonischen uit het signaal.

    signal: 1D array
    fs: samplefrequentie
    fundamental_freq: fundamentele frequentie (Hz)
    bw: half-width van bandstop rond elke harmonische
    """
    N = len(signal)
    X = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    
    # Bereken harmonischen binnen Nyquist
    max_harm = int((fs/2)//fundamental_freq)
    harmonics = [fundamental_freq * n for n in range(1, max_harm+1)]
    
    # Masker: verwijder harmonischen ± bw
    mask = np.ones_like(X, dtype=bool)
    for h in harmonics:
        mask &= np.abs(freqs - h) > bw
    
    # Terug naar tijdsdomein
    y = np.fft.irfft(X * mask, n=N)
    return y, harmonics