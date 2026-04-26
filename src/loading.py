# # Gekopieerd van Lucas
# import os
# import pandas as pd

# def find_study_files(base_path, study_number, TIMESTAMPS, require_signals=False):
#     """
#     Searches for files in a specific SWITCH study folder based on the study number.

#     Parameters:
#         base_path (str): The base directory containing the SWITCH folders.
#         study_number (str): The study number to search for (e.g., "002").
#         require_signals (bool): If True, file names must also contain 'Signals'.

#     Returns:
#         dict: A dictionary containing time points as keys and their file paths as values.
#         Returns an empty dictionary if the study folder or files are not found.
#     """
#     # Format the folder name to search for
#     folder_name = f"SWITCH{study_number}"
#     folder_path = os.path.join(base_path, folder_name)

#     # Check if the study folder exists
#     if not os.path.isdir(folder_path):
#         print(f"Study folder '{folder_name}' not found.")
#         return {}

#     # # List of time points to search for
#     # time_points = ["preswitch", "t0", "t30", "t60", "t90", "t120", "t150", "t180", "t210", "followup1", "followup2","followup3", "followup4", "followup5", "followup6", "failure"]
    
#     # Dictionary to store the file paths for each time point
#     files_found = {}

#     # Search for files corresponding to each time point
#     for time_point in TIMESTAMPS:
#         file_found = False
#         for root, _, files in os.walk(folder_path):
#             for file in files:
#                 if time_point in file and (not require_signals or 'Signals' in file):
#                     files_found[time_point] = os.path.join(root, file)
#                     file_found = True
#                     break
#             if file_found:
#                 break

#         if not file_found:
#             files_found[time_point] = None

#     return files_found

# def load_fluxmed_data(fluxmed_files, TIMESTAMP):
#     # Filepath to fluxmed .txt file
#     file_path = fluxmed_files[TIMESTAMP]

#     # Read the file, skipping the metadata rows and setting column names explicitly
#     fluxmed_data = pd.read_csv(
#         file_path,
#         delimiter="\t",
#         skiprows=5,  # Skip metadata lines (adjust based on the file structure)
#         header=0,  # Use the first row of actual data as column names
#     )

#     # # Drop the first row that contains units
#     fluxmed_data = fluxmed_data[1:].reset_index(drop=True)  # Drop the first row and reset the index
#     # Replace ',' with '.' in the entire DataFrame
#     fluxmed_data = fluxmed_data.replace(',', '.', regex=True)
#     # Convert all columns to numeric
#     fluxmed_data = fluxmed_data.apply(pd.to_numeric, errors='coerce')

#     # # Extract signals as numpy arrays
#     time = fluxmed_data["Time"].to_numpy()
#     flow = fluxmed_data["Flow"].to_numpy()
#     volume = fluxmed_data["Volume"].to_numpy()
#     paw = fluxmed_data["Paw"].to_numpy()
#     pes = fluxmed_data["Pes"].to_numpy()
#     ptpulm = fluxmed_data["Ptpulm"].to_numpy()
#     pga = fluxmed_data["Pga"].to_numpy()
#     ptdiaf = fluxmed_data["Ptdiaf"].to_numpy()

#     fluxmed_arrays = {"time": time, "flow": flow, "volume": volume, "paw": paw, "pes": pes, "ptpulm": ptpulm, "pga": pga, "ptdiaf": ptdiaf}
#     return fluxmed_arrays


import pickle
import pathlib
from pathlib import Path


class PosixPathFixUnpickler(pickle.Unpickler):
    """
    Fix voor pickle-bestanden die op Mac zijn gemaakt
    en op Windows worden ingeladen.
    """
    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            return pathlib.PurePosixPath
        return super().find_class(module, name)


def load_synchronized_sequence(sync_path):
    """
    Laadt een gesynchroniseerd SWITCH-bestand.

    Parameters
    ----------
    sync_path : str of Path
        Pad naar het gesynchroniseerde pickle-bestand.

    Returns
    -------
    sequence
        Ingeladen sequence-object met o.a. continuous_data en eit_data.
    """
    sync_path = Path(sync_path)

    if not sync_path.exists():
        raise FileNotFoundError(f"Gesynchroniseerd bestand niet gevonden: {sync_path}")

    with open(sync_path, "rb") as f:
        sequence = PosixPathFixUnpickler(f).load()

    return sequence


def load_synchronized_pes(sequence, key="synchronized_pes"):
    """
    Haalt het gesynchroniseerde PES-signaal uit de sequence.

    Returns
    -------
    time : np.ndarray
    pes : np.ndarray
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
    eit_signal
        EIT signal object.
    eit_time : np.ndarray
        Tijdvector van EIT.
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


def crop_eit_to_window(eit_signal, t_start, t_end):
    """
    Knipt EIT-signaal op basis van tijdsvenster.
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