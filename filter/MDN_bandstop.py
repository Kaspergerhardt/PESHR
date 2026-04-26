import numpy as np
from scipy.signal import find_peaks
from scipy.signal import welch  
from scipy.signal import butter, sosfiltfilt

def compute_welch_spectrum(signal, fs=256, noverlap=None):
    """
    Bereken het Welch-spectrum van een signaal.
    
    Parameters
    ----------
    signal : Het input signaal.
    fs : Samplingfrequentie van het signaal (default=256 Hz).
    segments : Hoeveel segmenten welch toepast.
    noverlap : 
        Het aantal overlappende punten (default=None).
    
    Returns
    -------
    f : De frequenties van het spectrum.
    Pxx : De vermogensspectrale dichtheid.
    """
    length = len(signal)
    segments = len(signal)//7000
    if segments == 0: 
        segments = 1

    f, Pxx = welch(signal, fs=fs, nperseg=length/segments, noverlap=noverlap, window='hann', scaling='spectrum')
    return f, Pxx

def find_peak_properties_bandstop(freqs, mag, f_min=0.67, f_max=4.0, percent_of_peak=0.5):
    """
    Zoek de hoogste piek binnen een gegeven frequentieband en bereken fundamentele frequentie,
    amplitude en piekbreedte bij een bepaalde drempel (percent_of_peak).

    Parameters
    ----------
    freqs : Frequencies van het spectrum.
    mag : Magnitude van het spectrum.
    f_min : Minimum frequentie voor zoekgebied (Hz).
    f_max : Maximum frequentie voor zoekgebied (Hz).
    percent_of_peak : Percentage van piekamplitude om piekbreedte te definiëren.

    Returns
    -------
    fundamental_freq : 
        Frequentie van de hoogste piek binnen de band.
    fundamental_amp : 
        Amplitude van de piek.
    peak_width : 
        Breedte van de piek bij amplitude ≤ percent_of_peak * piekamplitude.
    left_freq : 
        Linkergrens van piek bij threshold.
    right_freq : 
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
    # peak_idx, _ = find_peaks(mag_band)
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


def find_closest_peak_in_welch_spectrum(signal, fundamental_freq, fs=256, f_min=0.67, f_max=4.0, percent_of_peak=0.5):
    """
    Zoek de dichtstbijzijnde piek in het Welch-spectrum van het volledige signaal
    ten opzichte van de gevonden fundamentele frequentie van het breathhold-signaal.
    
    Parameters
    ----------
    signal : 
        Het input signaal (breathhold signaal).
    fundamental_freq : 
        De fundamentele frequentie gevonden in het breathhold-signaal.
    fs : 
        Sample frequentie van het signaal. (default=256 Hz).
    f_min :Minimale frequentie voor het zoekgebied. (default=0.8 Hz).
    f_max :
        Maximale frequentie voor het zoekgebied. (default=4.0 Hz).
    percent_of_peak : 
        Percentage van piekamplitude om piekbreedte te definiëren (default=0.5).
    nperseg :  
        Het aantal punten per segment voor de Welch berekening.
    
    Returns
    -------
    fundamental_freq_welch : 
        De fundamentele frequentie gevonden in het Welch-spectrum.
    peak_amp : 
        Amplitude van de piek.
    peak_width : 
        Breedte van de piek bij threshold.
    left_freq : 
        Linkergrens van de piek bij threshold.
    right_freq : 
        Rechtergrens van de piek bij threshold.
    """
    # Zorg ervoor dat noverlap 50% van nperseg is
    nperseg = fs * 10  # Bijvoorbeeld 4 seconden per segment
    noverlap = nperseg // 2  # 50% overlap
    segments = len(signal) // 7000
    # Bereken het Welch-spectrum met 50% overlap
    fwelch, Pxx = compute_welch_spectrum(signal, fs=256)
    
    # Zoek pieken in het spectrum
    peaks, _ = find_peaks(Pxx)
    
    if len(peaks) == 0:
        raise ValueError("Geen pieken gevonden in het spectrum.")
    
    # Zoek de piek die het dichtst bij de fundamentele frequentie ligt
    peak_freqs = fwelch[peaks]
    peak_mags = Pxx[peaks]
    
    # Vind de piek die het dichtst bij de fundamentele frequentie ligt
    dist_to_fundamental = np.abs(peak_freqs - fundamental_freq)
    closest_peak_idx = np.argmin(dist_to_fundamental)
    
    # De fundamentele frequentie in het spectrum
    fundamental_freq_welch = peak_freqs[closest_peak_idx]
    if fundamental_freq_welch < f_min or fundamental_freq_welch > f_max:
        fundamental_freq_welch = fundamental_freq  # Fallback naar oorspronkelijke frequentie
        print("Waarschuwing: Gevonden fundamentele frequentie buiten bereik, gebruik fundamentele frequentie uit breathhold.")
    fundamental_amp = peak_mags[closest_peak_idx]
    
    # Zoek naar de indexen van de frequenties die onder de drempel liggen
    threshold = percent_of_peak * fundamental_amp

    # Zoek de linkergrens (van de piek naar links, totdat we onder de drempel komen)    
    left_idx = peaks[closest_peak_idx]
    while left_idx > 0 and Pxx[left_idx] > threshold:
        left_idx -= 1
    left_freq = fwelch[left_idx]
    if left_freq < f_min:
        left_freq = f_min

    # Zoek de rechtergrens (van de piek naar rechts, totdat we onder de drempel komen)
    right_idx = peaks[closest_peak_idx]
    while right_idx < len(Pxx) - 1 and Pxx[right_idx] > threshold:
        right_idx += 1
    right_freq = fwelch[right_idx]

    # Bereken de piekbreedte
    peak_width = right_freq - left_freq
    
    return fundamental_freq_welch, fundamental_amp, peak_width, left_freq, right_freq, fwelch, Pxx

def normalize_to_peak(Pxx):
    """
    Normaliseer het spectrum zodat de piek gelijk is aan 1.
    
    Parameters
    ----------
    Pxx : 
        Het berekende spectrum (Power Spectral Density).
        
    Returns
    -------
    Pxx_normalized : 
        Het genormaliseerde spectrum waarbij de piek gelijk is aan 1.
    """
    max_amp = np.max(Pxx)  # Vind de maximale amplitude
    Pxx_normalized = Pxx / max_amp  # Deel alle waarden door de maximale amplitude
    return Pxx_normalized

def butter_bandstop(cutoff_low, cutoff_high, fs, order=5):
    """
    Genereer een Butterworth bandstopfilter.
    
    Parameters
    ----------
    cutoff_low : 
        Lage cutoff frequentie van het bandstopfilter.
    cutoff_high : 
        Hoge cutoff frequentie van het bandstopfilter.
    fs : 
        Samplingfrequentie.
    order : 
        Orde van het Butterworth filter.
    
    Returns
    -------
    sos : 
        Filtercoëfficiënten.
    """
    nyquist = 0.5 * fs
    low = cutoff_low / nyquist
    high = cutoff_high / nyquist
    if low <= 0 or high >= 1 or low >= high:
        print(low, high)
        raise ValueError("De cutofffrequenties moeten tussen 0 en 1 liggen en low < high zijn.")

    sos = butter(order, [low, high], btype='bandstop',output='sos')
    return sos

def remove_fundamental_and_harmonics(signal, fundamental_freq, fs=256, bw=0.01):
    """
    Verwijder de fundamentele frequentie en haar harmonischen uit het signaal met behulp van
    5e orde Butterworth bandstopfilters.
    
    Parameters
    ----------
    signal : 
        Het input signaal.
    fs : 
        Samplingfrequentie van het signaal.
    fundamental_freq : 
        De fundamentele frequentie van het signaal.
    bw : 
        Half-width van het bandstop rond elke harmonische frequentie.
    
    Returns
    -------
    signals : 
        Het gefilterde signaal met de fundamentele frequentie en harmonischen verwijderd.
    harmonics :De harmonischen die uit het signaal zijn verwijderd.
    """
    N = len(signal)
    
    # Bereken de harmonischen binnen Nyquist
    max_harm = int(((fs / 2)-1) // (fundamental_freq))
    harmonics = [fundamental_freq * n for n in range(1, max_harm + 1)]
    
    # Toepassen van de bandstopfilter op elke harmonische
    for h in harmonics:
        cutoff_low = h - bw / 2
        cutoff_high = h + bw / 2
        sos = butter_bandstop(cutoff_low, cutoff_high, fs)
        
        # Pas het bandstopfilter toe in het frequentiedomein
        signal = sosfiltfilt(sos, signal)
    return signal, harmonics