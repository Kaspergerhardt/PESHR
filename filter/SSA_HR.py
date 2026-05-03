import numpy as np
import pandas as pd
from scipy import signal

from scipy.signal import butter, filtfilt, find_peaks
from scipy.linalg import svd
from preprocessing import butter_bandpass_filter

# ------------------
# SSA functies
# ------------------

def ssa_decompose(x, fs, min_cycles=2):
    """
    Voert Singular Spectrum Analysis (SSA) uit op een 1D-signaal.

    Parameters
    ----------
    x : np.ndarray
        Ingangssignaal.
    L : int
        Vensterlengte voor de trajectmatrix.

    Returns
    -------
    U : np.ndarray
        Linker singuliere vectoren.
    s : np.ndarray
        Singuliere waarden.
    Vt : np.ndarray
        Getransponeerde rechter singuliere vectoren.
    L : int
        Gebruikte vensterlengte.
    K : int
        Aantal kolommen van de trajectmatrix.
    """
    # Vensterlengte
    L = int(fs/0.67)*min_cycles

    N = len(x)
    K = N - L + 1

    # Construeer de trajectmatrix uit overlappende segmenten
    X = np.column_stack([x[i:i+L] for i in range(K)])

    # Pas SVD toe op de trajectmatrix
    U, s, Vt = svd(X, full_matrices=False)

    return U, s, Vt, L, K


def diagonal_averaging(Xi):
    """
    Reconstrueert een 1D-signaal uit een SSA-componentmatrix
    via diagonale middeling.

    Parameters
    ----------
    Xi : np.ndarray
        Componentmatrix van één SSA-component.

    Returns
    -------
    np.ndarray
        Gereconstrueerd tijdsignaal.
    """
    # Bepaal matrixafmetingen en lengte van het uitgangssignaal
    L, K = Xi.shape
    N = L + K - 1

    # Initialiseer sommen en aantallen per anti-diagonaal
    sums = np.zeros(N)
    counts = np.zeros(N)

    # Tel bijdragen per anti-diagonaal op
    for i in range(L):
        sums[i:i+K] += Xi[i, :]
        counts[i:i+K] += 1

    return sums / counts

def reconstruct_selected_rcs(U, s, Vt, indices):
    """
    Reconstrueert geselecteerde SSA-componenten.

    Parameters
    ----------
    U : np.ndarray
        Linker singuliere vectoren.
    s : np.ndarray
        Singuliere waarden.
    Vt : np.ndarray
        Getransponeerde rechter singuliere vectoren.
    indices : iterable
        Indexen van te reconstrueren componenten.

    Returns
    -------
    np.ndarray
        Gereconstrueerde componenten, één per rij.
    """
    RCs = []

    # Reconstrueer iedere geselecteerde component
    for i in indices:
        Xi = s[i] * np.outer(U[:, i], Vt[i, :])
        rc = diagonal_averaging(Xi)
        RCs.append(rc)

    return np.array(RCs)


# --------------------
# Energie bepaling
# --------------------

def band_energy(signal, fs, f_low, f_high):
    """
    Berekent de energie van een signaal binnen een geselecteerde frequentieband.

    Parameters
    ----------
    signal : array-like
        Invoersignaal.
    fs : float
        Samplingfrequentie van het signaal.
    f_low : float
        Ondergrens van de frequentieband.
    f_high : float
        Bovengrens van de frequentieband.

    Returns
    -------
    float
        Totale energie binnen de opgegeven frequentieband.
    """
    signal = np.asarray(signal)
    n = len(signal)

    # Leeg signaal bevat geen energie
    if n == 0:
        return 0.0

    # Verwijder DC-component en pas Hann-window toe
    x = signal - np.mean(signal)
    w = np.hanning(n)
    xw = x * w

    # Bereken frequentiespectrum
    freqs = np.fft.rfftfreq(n, d=1/fs)
    spec = np.abs(np.fft.rfft(xw)) ** 2

    # Selecteer frequenties binnen de gewenste band
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0

    return float(np.sum(spec[mask]))


def signal_energy(x):
    """
    Berekent de energie van een tijdsignaal in het tijddomein.
    """
    x = np.asarray(x)
    if len(x) == 0:
        return 0.0
    x = x - np.mean(x)
    return float(np.sum(x ** 2))


def rc_pair_contribution(signal, RCs, pair):
    """
    Berekent de klinische impact van een paar SSA-componenten (RCs)
    op een signaal door te kijken naar de verandering in amplitude.

    Parameters
    ----------
    signal : array-like
        Originele signaal.
    RCs : np.ndarray
        Gereconstrueerde componenten (elke rij is een component).
    pair : iterable
        Indexen van de twee componenten die geëvalueerd worden.

    Returns
    -------
    float
        Verschil in piek-tot-piek amplitude (ptp) tussen het originele
        en het 'gedenoisede' signaal. Een hogere waarde betekent dat
        het componentpaar meer invloed had op het signaal.
    """
    # Combineer de geselecteerde componenten
    x_pair = np.sum(RCs[list(pair)], axis=0)

    # Verwijder deze componenten uit het originele signaal
    denoised = signal - x_pair

    # Bereken piek-tot-piek amplitudes
    delta_raw = np.ptp(signal)
    delta_denoised = np.ptp(denoised)

    # Impact = verschil in amplitude
    return delta_raw - delta_denoised

def extract_cgo_ssa(
    signal,
    fs,
    min_cycles=2,
    max_components=10,
    cardiac_band=(0.67, 4.0),
    respiratory_band=(0.17, 0.80),
    resp_penalty=5.0,
):
    """
    Extraheert automatisch een CGO-signaal uit een ingangssignaal met SSA.

    Selectie gebeurt per SSA RC-paar op basis van:
        score = cardiac_energy - resp_penalty * respiratory_energy

    Parameters
    ----------
    signal : np.ndarray
    fs : float
    max_components : int
    cardiac_band : tuple
    respiratory_band : tuple
    resp_penalty : float
        Strafgewicht voor respiratoire energie.
        Hogere waarde = respiratoire componenten harder afstraffen.

    Returns
    -------
    cgo : np.ndarray
    RCs : np.ndarray
    valid_pairs : list of tuple
    chosen_pair : tuple
    s : np.ndarray
    pair_scores : list of dict
    """
    eps = 1e-12

    U, s, Vt, L, K = ssa_decompose(signal, fs, min_cycles)

    n_comp = min(max_components, len(s))
    all_idx = list(range(n_comp))
    RCs = reconstruct_selected_rcs(U, s, Vt, all_idx)

    rc_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    pair_scores = []

    for pair in rc_pairs:
        rc_pair_signal = np.sum(RCs[list(pair)], axis=0)

        e_card = band_energy(rc_pair_signal, fs, cardiac_band[0], cardiac_band[1])
        e_resp = band_energy(rc_pair_signal, fs, respiratory_band[0], respiratory_band[1])

        score = e_card - resp_penalty * e_resp

        contribution = rc_pair_contribution(signal, RCs, pair)

        pair_scores.append({
            "pair": pair,
            "cardiac_energy": e_card,
            "respiratory_energy": e_resp,
            "score": score,
            "contribution": contribution,  
        })

    best_item = max(pair_scores, key=lambda x: x["score"])
    chosen_pair = best_item["pair"]

    for item in pair_scores:
        item["chosen"] = item["pair"] == chosen_pair

    cgo = np.sum(RCs[list(chosen_pair)], axis=0)

    return cgo, RCs, rc_pairs, chosen_pair, s, pair_scores


def rc_pair_overview_dataframe(pair_scores, RCs, rc_pairs, fs, hr_eit):
    """
    Overview dataframe per RC pair with:
    - CGO-based selection (score)
    - HR-EIT-based selection (frequency)
    """

    # ---- score info ----
    df_scores = pd.DataFrame(pair_scores).copy()

    if df_scores.empty:
        return pd.DataFrame(columns=[
            "pair",
            "cardiac_energy",
            "respiratory_energy",
            "score",
            "contribution",
            "f_dom_hz",
            "f_target_hz",
            "chosen_cgo",
            "chosen_hr_eit",
        ])

    score_cols = [
        "pair",
        "cardiac_energy",
        "respiratory_energy",
        "score",
        "contribution",
    ]
    df_scores = df_scores[score_cols].copy()
    df_scores["pair"] = df_scores["pair"].apply(tuple)

    # ---- frequency info ----
    f_target = hr_eit
    freq_rows = []

    for pair in rc_pairs:
        pair = tuple(pair)

        x = np.sum(RCs[list(pair)], axis=0)
        x = x - np.mean(x)

        freqs = np.fft.rfftfreq(len(x), d=1 / fs)
        psd = np.abs(np.fft.rfft(x)) ** 2

        f_dom = freqs[np.argmax(psd)]
        freq_diff = abs(f_dom - f_target)

        freq_rows.append({
            "pair": pair,
            "f_dom_hz": f_dom,
            "f_target_hz": f_target,
            "freq_diff": freq_diff,
        })

    df_freq = pd.DataFrame(freq_rows)

    # ---- merge ----
    df = df_scores.merge(df_freq, on="pair", how="inner")

    # ---- selections ----
    # CGO: highest score
    chosen_cgo_pair = df.loc[df["score"].idxmax(), "pair"]

    # HR EIT: closest frequency
    chosen_hr_eit_pair = df.loc[df["freq_diff"].idxmin(), "pair"]

    df["chosen_cgo"] = df["pair"] == chosen_cgo_pair
    df["chosen_hr_eit"] = df["pair"] == chosen_hr_eit_pair

    # ---- final formatting ----
    df = df[
        [
            "pair",
            "cardiac_energy",
            "respiratory_energy",
            "score",
            "contribution",
            "f_dom_hz",
            "f_target_hz",
            "chosen_cgo",
            "chosen_hr_eit",
        ]
    ].sort_values("score", ascending=False).reset_index(drop=True)

    return df



# --------------------
# Piekdetectie
# --------------------

def compute_hr_from_peaks(cgo, time, fs, min_bpm=40, max_bpm=240):
    """
    Schat de hartfrequentie op basis van piekdetectie in het CGO-signaal.

    Parameters
    ----------
    cgo : np.ndarray, CGO-signaal.
    time : np.ndarray
        Tijdvector behorend bij het signaal.
    fs : float
        Samplingfrequentie in Hz.
    min_bpm : float, optional
        Ondergrens voor plausibele HR in bpm.
    max_bpm : float, optional
        Bovengrens voor plausibele HR in bpm.

    Returns
    -------
    t_hr : np.ndarray
        Tijdstippen van de HR-schattingen.
    hr : np.ndarray
        Geschatte hartfrequentie in bpm.
    peaks : np.ndarray
        Indexen van de gedetecteerde pieken.
    """
    # Minimale piekafstand op basis van maximale hartfrequentie
    min_distance = int(fs * 60 / max_bpm)
    
    # Prominentiedrempel
    prominence = 0.1 * np.std(cgo)

    # Detecteer pieken in het signaal
    peaks, props = find_peaks(cgo, distance=min_distance, prominence=prominence)

    # Zonder minstens twee pieken kan geen HR worden berekend
    if len(peaks) < 2:
        return np.array([]), np.array([]), peaks

    # Zet piekindexen om naar tijdstippen
    peak_times = time[peaks]

    # Bereken inter-beat intervals en HR
    ibi = np.diff(peak_times)
    hr = 60 / ibi

    # Plaats HR op het midden tussen twee opeenvolgende pieken
    t_hr = peak_times[:-1] + ibi / 2

    # Behoud alleen fysiologisch plausibele waarden
    valid = (hr >= min_bpm) & (hr <= max_bpm)
    return t_hr[valid], hr[valid], peaks


# ------------------------
# Verwerking Pes signaal
# ------------------------


def process_pes(time_window, pes_window, fs):
    """
    Verwerkt een PES-signaalvenster tot een CGO-reconstructie,
    piekgebaseerde HR en bijhorende SSA-score-informatie.
    """
    pes_bp = butter_bandpass_filter(pes_window, fs, 0.67, 4.0)

    cgo, RCs, rc_pairs, chosen_pair, s, pair_scores = extract_cgo_ssa(
        pes_bp,
        fs,
        max_components=10,
    )

    pes_bp_energy = signal_energy(pes_bp)
    cgo_energy = signal_energy(cgo)
    cgo_to_pesbp_ratio = cgo_energy / (pes_bp_energy + 1e-12)

    t_hr, hr, peaks = compute_hr_from_peaks(cgo, time_window, fs)

    return (
        pes_bp,
        cgo,
        RCs,
        rc_pairs,
        chosen_pair,
        t_hr,
        hr,
        peaks,
        s,
        pair_scores,
        cgo_energy,
        pes_bp_energy,
        cgo_to_pesbp_ratio,
    )










