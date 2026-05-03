from scipy.signal import butter, filtfilt
import numpy as np

def butter_bandpass_filter(signal, fs, lowcut, highcut, order=4):
    """
    Filtert een signaal met een Butterworth-banddoorlaatfilter.

    Parameters
    ----------
    signal : np.ndarray
        Ingangssignaal.
    fs : float
        Samplingfrequentie in Hz.
    lowcut : float
        Onderste afsnijfrequentie in Hz.
    highcut : float
        Bovenste afsnijfrequentie in Hz.
    order : int, optional
        Filterorde.

    Returns
    -------
    np.ndarray
        Bandgefilterd signaal.
    """
    nyq = 0.5 * fs
    b,a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)
