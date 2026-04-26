import numpy as np
from scipy.signal import find_peaks, savgol_filter


# ------------------
# NLMS
# ------------------

def nlms_window(x, t, f0, M=1, A=10, L=30, mu=0.002, w_init=None):
    """
    Past een NLMS-filter toe op een signaalvenster met een harmonische referentie.

    Parameters
    ----------
    x : np.ndarray
        Ingangssignaal.
    t : np.ndarray
        Tijdvector behorend bij het signaal.
    f0 : float
        Referentiefrequentie in Hz.
    M : int, optional
        Aantal harmonischen in de referentie.
    A : float, optional
        Amplitude van de referentiesinus.
    L : int, optional
        Filterlengte.
    mu : float, optional
        Adaptatiestap van de NLMS.
    w_init : np.ndarray or None, optional
        Initiële filtergewichten.

    Returns
    -------
    y : np.ndarray
        Geschatte referentiecomponent.
    e : np.ndarray
        Residusignaal.
    w : np.ndarray
        Geüpdatete filtergewichten.
    """
    ref = np.zeros_like(x, dtype=float)

    for h in range(1, M + 1):
        ref += A * np.sin(2 * np.pi * h * f0 * t)

    if w_init is None:
        w = np.zeros(L, dtype=float)
    else:
        w = np.asarray(w_init, dtype=float).copy()

    y = np.zeros(len(x), dtype=float)
    e = np.zeros(len(x), dtype=float)

    for k in range(L, len(x)):
        xv = ref[k - L:k][::-1]
        y[k] = np.dot(w, xv)
        e[k] = x[k] - y[k]
        w += mu * e[k] * xv / (np.dot(xv, xv) + 1e-6)

    return y, e, w


# ------------------
# Smoothing
# ------------------

def smooth_signal(x, fs, smooth_sec=0.08, poly=2):
    """
    Maakt een signaal glad met een Savitzky-Golay filter.

    Parameters
    ----------
    x : np.ndarray
        Ingangssignaal.
    fs : float
        Samplingfrequentie in Hz.
    smooth_sec : float, optional
        Vensterlengte in seconden.
    poly : int, optional
        Polynoomorde.

    Returns
    -------
    np.ndarray
        Gladgestreken signaal.
    """
    win = max(5, int(smooth_sec * fs))

    if win % 2 == 0:
        win += 1

    if win >= len(x):
        return x.copy()

    poly = min(poly, win - 1)
    return savgol_filter(x, window_length=win, polyorder=poly)


# ------------------
# Stabiel deel
# ------------------

def stable_slice(n, fs, edge_sec=1.0, L=30):
    """
    Bepaalt het stabiele middenstuk van een signaalvenster.

    Parameters
    ----------
    n : int
        Lengte van het signaal.
    fs : float
        Samplingfrequentie in Hz.
    edge_sec : float, optional
        Tijd aan beide randen die wordt uitgesloten.
    L : int, optional
        Minimale marge gelijk aan filterlengte.

    Returns
    -------
    slice
        Slice-object voor het stabiele gedeelte.
    """
    edge = max(int(edge_sec * fs), L)

    if 2 * edge >= n:
        return slice(0, n)

    return slice(edge, n - edge)


# ------------------
# Piekdetectie en HR
# ------------------

def compute_hr_from_cgo(cgo, time, fs, bpm=(40, 180),
                        prominence=0.35, smooth_sec=0.08,
                        edge_sec=1.0, L=30):
    """
    Detecteert pieken in het CGO-signaal en schat de hartfrequentie.

    Parameters
    ----------
    cgo : np.ndarray
        CGO-signaal.
    time : np.ndarray
        Tijdvector.
    fs : float
        Samplingfrequentie in Hz.
    bpm : tuple, optional
        Minimale en maximale plausibele hartfrequentie.
    prominence : float, optional
        Relatieve prominentiedrempel.
    smooth_sec : float, optional
        Smoothingvenster in seconden.
    edge_sec : float, optional
        Randgedeelte dat niet gebruikt wordt voor piekdetectie.
    L : int, optional
        Filterlengte, gebruikt voor stabiele slice.

    Returns
    -------
    cgo_s : np.ndarray
        Gladgestreken CGO-signaal.
    peaks : np.ndarray
        Piekindexen in het volledige venster.
    peak_t : np.ndarray
        Tijdstippen van de pieken.
    t_hr : np.ndarray
        Tijdstippen van HR-schattingen.
    hr : np.ndarray
        Hartfrequentie in bpm.
    f_med : float or None
        Mediane hartfrequentie in Hz.
    """
    stable = stable_slice(len(cgo), fs, edge_sec=edge_sec, L=L)

    cgo_mid = cgo[stable]
    time_mid = time[stable]

    cgo_s_mid = smooth_signal(cgo_mid, fs, smooth_sec=smooth_sec)

    min_bpm, max_bpm = bpm
    min_dist = max(1, int(fs * 60 / max_bpm))
    prom = prominence * np.std(cgo_s_mid)

    peaks_mid, _ = find_peaks(cgo_s_mid, distance=min_dist, prominence=prom)
    peak_t = time_mid[peaks_mid]

    peak_offset = stable.start if stable.start is not None else 0
    peaks = peaks_mid + peak_offset

    cgo_s = cgo.copy()
    cgo_s[stable] = cgo_s_mid

    if len(peak_t) < 2:
        return cgo_s, peaks, peak_t, np.array([]), np.array([]), None

    ibi = np.diff(peak_t)
    ibi = ibi[ibi > 0]

    if len(ibi) == 0:
        return cgo_s, peaks, peak_t, np.array([]), np.array([]), None

    hr = 60 / ibi
    t_hr = peak_t[:-1] + ibi / 2

    ok = (hr >= min_bpm) & (hr <= max_bpm)

    if not np.any(ok):
        return cgo_s, peaks, peak_t, np.array([]), np.array([]), None

    hr_valid = hr[ok]
    t_hr_valid = t_hr[ok]
    f_med = np.median(hr_valid) / 60.0

    return cgo_s, peaks, peak_t, t_hr_valid, hr_valid, f_med


# ------------------
# NLMS frequentiescan
# ------------------

def scan_f0_nlms(x, t, f0_center, span_bpm=40, n_grid=41,
                 M=1, A=10, L=30, mu=0.002):
    """
    Scant frequenties rond een referentie en bepaalt de beste NLMS-frequentie
    op basis van minimale residu-MSE.

    Parameters
    ----------
    x : np.ndarray
        Ingangssignaal.
    t : np.ndarray
        Tijdvector.
    f0_center : float
        Centrale frequentie in Hz.
    span_bpm : float, optional
        Breedte van het scanbereik in bpm.
    n_grid : int, optional
        Aantal frequentiepunten.
    M : int, optional
        Aantal harmonischen.
    A : float, optional
        Referentie-amplitude.
    L : int, optional
        Filterlengte.
    mu : float, optional
        Adaptatiestap.

    Returns
    -------
    f_grid : np.ndarray
        Gescande frequenties in Hz.
    mse : np.ndarray
        MSE per frequentie.
    best_f : float
        Beste frequentie in Hz.
    sharpness : float
        Maat voor scherpte/confidence van de scan.
    """
    if f0_center is None:
        return None, None, None, None

    span_hz = span_bpm / 60.0
    f_grid = np.linspace(f0_center - span_hz / 2, f0_center + span_hz / 2, n_grid)

    mse = []

    for f in f_grid:
        _, e, _ = nlms_window(x, t, f, M=M, A=A, L=L, mu=mu, w_init=None)
        mse.append(np.mean(e[L:] ** 2))

    mse = np.array(mse)
    best_f = f_grid[np.argmin(mse)]
    sharpness = np.std(mse) / (np.mean(mse) + 1e-8)

    return f_grid, mse, best_f, sharpness


# ------------------
# Verwerking 1 venster
# ------------------

def process_pes_nlms(time_window, pes_window, fs, f0,
                     M=1, A=10, L=30, mu=0.002,
                     bpm=(40, 180),
                     prominence=0.35,
                     smooth_sec=0.08,
                     edge_sec=1.0,
                     scan_span_bpm=20,
                     scan_grid=41,
                     w_init=None):
    """
    Verwerkt één PES-signaalvenster met NLMS, piekdetectie en frequentiescan.

    Parameters
    ----------
    time_window : np.ndarray
        Tijdvector van het venster.
    pes_window : np.ndarray
        PES-signaal in het venster.
    fs : float
        Samplingfrequentie in Hz.
    f0 : float
        Vaste referentiefrequentie in Hz.
    M, A, L, mu : various, optional
        NLMS-parameters.
    bpm : tuple, optional
        Plausibel HR-bereik in bpm.
    prominence : float, optional
        Relatieve prominentie voor piekdetectie.
    smooth_sec : float, optional
        Smoothing in seconden.
    edge_sec : float, optional
        Randuitsluiting in seconden.
    scan_span_bpm : float, optional
        Scanbreedte rond f0 in bpm.
    scan_grid : int, optional
        Aantal scanpunten.
    w_init : np.ndarray or None, optional
        Starttoestand van NLMS-gewichten.

    Returns
    -------
    x : np.ndarray
        Origineel PES-venster.
    cgo : np.ndarray
        NLMS-uitgang.
    res : np.ndarray
        Residusignaal.
    cgo_s : np.ndarray
        Gladgestreken CGO.
    peaks : np.ndarray
        Piekindexen.
    peak_t : np.ndarray
        Piektijden.
    t_hr : np.ndarray
        Tijdstippen van HR-schattingen.
    hr : np.ndarray
        HR-waarden in bpm.
    f_peak : float or None
        Mediane HR-frequentie uit pieken in Hz.
    f_grid : np.ndarray
        Frequentierooster van de NLMS-scan.
    mse : np.ndarray
        MSE per frequentie.
    f_nlms : float
        Beste NLMS-frequentie in Hz.
    sharp : float
        Confidence-maat van de scan.
    w_state : np.ndarray
        Eindtoestand van NLMS-gewichten.
    """
    cgo, res, w_state = nlms_window(
        pes_window, time_window, f0,
        M=M, A=A, L=L, mu=mu, w_init=w_init
    )

    cgo_s, peaks, peak_t, t_hr, hr, f_peak = compute_hr_from_cgo(
        cgo, time_window, fs,
        bpm=bpm,
        prominence=prominence,
        smooth_sec=smooth_sec,
        edge_sec=edge_sec,
        L=L
    )

    f_grid, mse, f_nlms, sharp = scan_f0_nlms(
        pes_window, time_window, f0,
        span_bpm=scan_span_bpm,
        n_grid=scan_grid,
        M=M, A=A, L=L, mu=mu
    )

    return (
        pes_window,
        cgo,
        res,
        cgo_s,
        peaks,
        peak_t,
        t_hr,
        hr,
        f_peak,
        f_grid,
        mse,
        f_nlms,
        sharp,
        w_state
    )


# ------------------
# Analyse over tijd
# ------------------

def nlms_hour_analysis(pes, time, fs, f0_init_func,
                       win_sec=10, step_sec=60, max_minutes=60,
                       bpm=(40, 180),
                       M=1, A=10, L=30, mu=0.002,
                       prominence=0.35, smooth_sec=0.08, edge_sec=1.0,
                       scan_span_bpm=20, scan_grid=41):
    """
    Voert NLMS-analyse uit op opeenvolgende vensters van een PES-signaal.

    Parameters
    ----------
    pes : np.ndarray
        Volledig PES-signaal.
    time : np.ndarray
        Volledige tijdvector.
    fs : float
        Samplingfrequentie in Hz.
    f0_init_func : callable
        Functie die een initiële f0 in Hz teruggeeft.
    win_sec : float, optional
        Vensterlengte in seconden.
    step_sec : float, optional
        Stapgrootte tussen vensters in seconden.
    max_minutes : float, optional
        Maximale analyseperiode in minuten.
    bpm, M, A, L, mu, prominence, smooth_sec, edge_sec, scan_span_bpm, scan_grid
        Analyseparameters.

    Returns
    -------
    results : list of dict
        Resultaten per venster.
    """
    n_win = int(win_sec * fs)
    n_step = int(step_sec * fs)

    f0 = f0_init_func(pes[:n_win], time[:n_win], fs)

    results = []
    w_state = None

    for i, start in enumerate(range(0, len(pes) - n_win, n_step)):
        current_time = i * step_sec

        if current_time > max_minutes * 60:
            break

        stop = start + n_win
        x = pes[start:stop]
        t = time[start:stop]

        (
            x_out,
            cgo,
            res,
            cgo_s,
            peaks,
            peak_t,
            t_hr,
            hr,
            f_peak,
            f_grid,
            mse,
            f_nlms,
            sharp,
            w_state
        ) = process_pes_nlms(
            time_window=t,
            pes_window=x,
            fs=fs,
            f0=f0,
            M=M,
            A=A,
            L=L,
            mu=mu,
            bpm=bpm,
            prominence=prominence,
            smooth_sec=smooth_sec,
            edge_sec=edge_sec,
            scan_span_bpm=scan_span_bpm,
            scan_grid=scan_grid,
            w_init=w_state
        )

        results.append({
            "time": current_time,
            "f0": f0,
            "f_peak": f_peak,
            "f_nlms": f_nlms,
            "sharp": sharp,
            "x": x_out,
            "res": res,
            "cgo": cgo,
            "cgo_s": cgo_s,
            "t": t,
            "peaks": peaks,
            "peak_t": peak_t,
            "t_hr": t_hr,
            "hr": hr,
            "mse": mse,
            "f_grid": f_grid,
            "w": w_state.copy() if w_state is not None else None
        })

    return results