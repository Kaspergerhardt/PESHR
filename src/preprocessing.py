from scipy.signal import butter, filtfilt
import numpy as np

def butter_lowpass_filter(signal, cutoff=5, fs=256, order=4):
    """
    Past een Butterworth low-pass filter toe op een signaal.
    
    Parameters
    ----------
    signal : array-like
        Het inputsignaal dat gefilterd moet worden.
    cutoff : float, optional
        De cutoff-frequentie van de filter in Hz. Default is 5 Hz.
    fs : float, optional
        Samplingfrequentie van het signaal in Hz. Default is 256 Hz.
    order : int, optional
        Orde van de Butterworth filter. Default is 4.
        
    Returns
    -------
    filtered_signal : numpy.ndarray
        Het gefilterde signaal.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal