import numpy as np
from scipy.signal import correlate

def estimate_delay(x_ref, x_delayed):
    """
    Schat delay tussen x_ref (origineel PES)
    en x_delayed (gefilterd signaal)
    """
    corr = correlate(x_delayed, x_ref, mode='full')
    lags = np.arange(-len(x_ref) + 1, len(x_ref))
    delay = lags[np.argmax(corr)]
    return delay

def nlms_filter(pes, time, fundamental_freq, M=10, A_m=10, L=150, mu=0.01):
    """
    NLMS-adaptief filter voor PES-signalen met CGO-ruis
    met automatische delay-correctie
    
    Parameters:
    -----------
    pes : 
        Het originele PES-signaal (1D array)
    time : 
        Tijdvector van hetzelfde formaat als pes
    fundamental_freq : 
        Basisfrequentie voor CGO referentie
    M : 
        Aantal harmonischen in CGO referentie
    A_m : 
        Amplitude van CGO
    L : 
        Filterlengte (aantal taps)
    mu : 
        LMS-leersnelheid
    delay_factor : 
        Factor van L om delay te compenseren (0.25 betekent L/4)

    Returns:
    --------
    pfilt : 
        Gefilterd PES-signaal, delay gecorrigeerd
    pfilt_raw : 
        Gefilterd PES-signaal zonder delay correctie
    w : 
        Filtercoëfficiënten
    x_train : 
        Gebruikte CGO referentie (pes + ns)
    """

    # --- Constructie CGO referentie ---
    ns = np.zeros_like(pes)
    for m in range(1, M+1):
        ns += A_m * np.sin(2*np.pi*m*fundamental_freq*time)

    x_train = pes + ns
    d_train = pes

    # --- NLMS-adaptief filter ---
    w = np.zeros(L)
    pfilt_raw = np.zeros(len(pes))

    for n in range(L, len(pes)):
        x_vec = x_train[n-L:n][::-1]
        y_n = np.dot(w, x_vec)
        e_n = d_train[n] - y_n
        w += mu * e_n * x_vec / (np.dot(x_vec, x_vec) + 1e-6)

        p_vec = pes[n-L:n][::-1]
        pfilt_raw[n] = np.dot(w, p_vec)

    # --- Automatische delay schatting ---
    delay = estimate_delay(pes, pfilt_raw)

    print(f"Automatisch geschatte delay: {delay} samples")

    # --- Delay correctie ---
    pfilt = np.roll(pfilt_raw, -delay)

    if delay > 0:
        pfilt[-delay:] = pfilt_raw[-delay:]
    elif delay < 0:
        pfilt[:-delay] = pfilt_raw[:-delay]

    return pfilt, pfilt_raw, w, x_train, delay

def nlms_filter_manualdelay(pes, time, fundamental_freq, M=10, A_m=10, L=150, mu=0.01, delay_factor=0.25):
    """
    NLMS-adaptief filter voor PES-signalen met CGO-ruis.
    
    Parameters:
    -----------
    pes : 
        Het originele PES-signaal 
    time : 
        Tijdvector van hetzelfde formaat als pes
    fundamental_freq : 
        Basisfrequentie voor CGO referentie
    M : 
        Aantal harmonischen in CGO referentie
    A_m : 
        Amplitude van CGO
    L : 
        Filterlengte (aantal taps)
    mu : 
        LMS-leersnelheid
    delay_factor : 
        Factor van L om delay te compenseren (0.25 betekent L/4)

    Returns:
    --------
    pfilt : 
        Gefilterd PES-signaal, delay gecorrigeerd
    pfilt_raw : 
        Gefilterd PES-signaal zonder delay correctie
    w : 
        Filtercoëfficiënten
    x_train : 
        Gebruikte CGO referentie (pes + ns)
    """
    # --- Constructie CGO referentie ---
    ns = np.zeros_like(pes)
    for m in range(1, M+1):
        ns += A_m * np.sin(2*np.pi*m*fundamental_freq*time)

    x_train = pes + ns  # referentie
    d_train = pes       # hoofdchannel

    # --- NLMS-adaptief filter ---
    w = np.zeros(L)
    pfilt_raw = np.zeros(len(pes))
    for n in range(L, len(pes)):
        x_vec = x_train[n-L:n][::-1]
        y_n = np.dot(w, x_vec)
        e_n = d_train[n] - y_n
        w += mu * e_n * x_vec / (np.dot(x_vec, x_vec) + 1e-6)  # NLMS update
        p_vec = pes[n-L:n][::-1]
        pfilt_raw[n] = np.dot(w, p_vec)

    # --- Delay compensatie ---
    delay = int(L * delay_factor)
    pfilt = np.roll(pfilt_raw, -delay)
    pfilt[-delay:] = pfilt_raw[-delay:]

    return pfilt, pfilt_raw, w, x_train

def nlms_filter_fwbw(pes, time, fundamental_freq, M=10, A_m=10, L=150, mu=0.01):

    ns = np.zeros_like(pes)
    for m in range(1, M+1):
        ns += A_m * np.sin(2*np.pi*m*fundamental_freq*time)

    x_train = pes + ns
    d_train = pes

    # --- Forward NLMS ---
    w = np.zeros(L)
    pfwd = np.zeros(len(pes))

    for n in range(L, len(pes)):
        x_vec = x_train[n-L:n][::-1]
        y_n = np.dot(w, x_vec)
        e_n = d_train[n] - y_n

        w += mu * e_n * x_vec / (np.dot(x_vec, x_vec) + 1e-6)

        p_vec = pes[n-L:n][::-1]
        pfwd[n] = np.dot(w, p_vec)

    # --- Backward filtering (fixed weights) ---
    pbwd = np.zeros_like(pfwd)
    pfwd_rev = pfwd[::-1]

    for n in range(L, len(pfwd_rev)):
        p_vec = pfwd_rev[n-L:n][::-1]
        pbwd[n] = np.dot(w, p_vec)

    pfilt = pbwd[::-1]

    return pfilt, pfwd, w, x_train
