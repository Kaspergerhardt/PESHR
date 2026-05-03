import pickle
import pathlib
from pathlib import Path


class PosixPathFixUnpickler(pickle.Unpickler):
    """
    Fix voor pickle-bestanden die op Mac/Linux zijn gemaakt
    en op Windows worden ingeladen.
    """
    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            return pathlib.PurePosixPath
        return super().find_class(module, name)


def load_synchronized_sequence(sync_path):
    """
    Laadt een gesynchroniseerd SWITCH-bestand.
    """
    sync_path = Path(sync_path)

    if not sync_path.exists():
        raise FileNotFoundError(f"Gesynchroniseerd bestand niet gevonden: {sync_path}")

    with open(sync_path, "rb") as f:
        return PosixPathFixUnpickler(f).load()


def load_synchronized_pes(sequence, key="synchronized_pes"):
    """
    Haalt het gesynchroniseerde PES-signaal uit de sequence.

    Returns
    -------
    time, pes
    """
    if key not in sequence.continuous_data:
        raise KeyError(f"'{key}' niet gevonden in sequence.continuous_data")

    pes_signal = sequence.continuous_data[key]

    return pes_signal.time, pes_signal.values


def load_synchronized_eit(sequence, key="raw"):
    """
    Haalt het EIT-signaal uit de sequence.

    Returns
    -------
    eit_signal, eit_time
    """
    if key not in sequence.eit_data:
        raise KeyError(f"'{key}' niet gevonden in sequence.eit_data")

    eit_signal = sequence.eit_data[key]

    return eit_signal, eit_signal.time


def get_window_mask(time, t_start, t_end):
    """
    Maakt een boolean mask voor een tijdsvenster.
    """
    return (time >= t_start) & (time <= t_end)


def crop_pes_to_window(time, pes, t_start, t_end):
    """
    Knipt PES-signaal op basis van tijdsvenster.

    Returns
    -------
    time_window, pes_window
    """
    mask = get_window_mask(time, t_start, t_end)

    time_window = time[mask]
    pes_window = pes[mask]

    if len(time_window) == 0:
        raise ValueError(
            f"Geen PES-data gevonden tussen {t_start:.2f}s en {t_end:.2f}s"
        )

    return time_window, pes_window


def crop_eit_to_window(eit_signal, t_start, t_end):
    """
    Knipt EIT-signaal op basis van tijdsvenster.

    Returns
    -------
    eit_window, start_idx, end_idx
    """
    eit_time = eit_signal.time
    mask = get_window_mask(eit_time, t_start, t_end)
    idx = mask.nonzero()[0]

    if len(idx) == 0:
        raise ValueError(
            f"Geen EIT-data gevonden tussen {t_start:.2f}s en {t_end:.2f}s"
        )

    start_idx = idx[0]
    end_idx = idx[-1] + 1

    eit_window = eit_signal[start_idx:end_idx]

    return eit_window, start_idx, end_idx
