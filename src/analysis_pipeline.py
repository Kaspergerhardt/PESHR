import pickle
import pathlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("..").resolve() / "eitprocessing"))
from eitprocessing.features.rate_detection import RateDetection

sys.path.append(r"../src")
from preprocessing import butter_lowpass_filter

sys.path.append(r"../filter")
from SSA_HR import process_pes, rc_pair_overview_dataframe

sys.path.append(r"../config")
from dataset_config import get_dataset_window


WANTED_COLUMNS = [
    "patient", "phase", "pair",
    "cardiac_energy", "respiratory_energy", "score", "clinical_impact",
    "f_dom_bpm", "f_target_bpm",
    "chosen_cgo", "chosen_hr_eit", "mismatch_rc_pair",
    "HR_CGO_bpm", "HR_EIT_bpm",
]


def round_numeric_columns(df, decimals=2):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(decimals)
    return df


def add_bpm_columns(df):
    df = df.copy()

    if "f_dom_hz" in df.columns:
        df["f_dom_bpm"] = df["f_dom_hz"] * 60

    if "f_target_hz" in df.columns:
        df["f_target_bpm"] = df["f_target_hz"] * 60

    return df


class PosixPathFixUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            return pathlib.PurePosixPath
        return super().find_class(module, name)


def load_sequence(sync_path):
    with open(sync_path, "rb") as f:
        return PosixPathFixUnpickler(f).load()


def get_pes_window(sequence, patient, phase, fs):
    pes_signal = sequence.continuous_data["synchronized_pes"]

    time = pes_signal.time
    pes = pes_signal.values

    t_start, t_end = get_dataset_window(patient, phase)
    t_start += time[0]
    t_end += time[0]

    pes_filt = butter_lowpass_filter(pes, cutoff=5, fs=fs)
    mask = (time >= t_start) & (time <= t_end)

    time_window = time[mask]
    pes_window = pes_filt[mask]

    if len(time_window) == 0:
        raise ValueError("leeg PES venster")

    return time_window, pes_window, t_start, t_end


def get_eit_window(sequence, t_start, t_end):
    eit_signal = sequence.eit_data["raw"]
    eit_time = eit_signal.time

    mask = (eit_time >= t_start) & (eit_time <= t_end)
    idx = np.where(mask)[0]

    if len(idx) == 0:
        raise ValueError("leeg EIT venster")

    return eit_signal[idx[0]:idx[-1] + 1]


def estimate_hr_eit(eit_window):
    rd = RateDetection(subject_type="adult")
    _, hr_eit_hz = rd.apply(eit_window, captures={})

    hr_eit_bpm = hr_eit_hz * 60 if pd.notna(hr_eit_hz) else np.nan

    return hr_eit_hz, hr_eit_bpm


def analyse_patient_phase(patient, phase, sync_base, fs):
    sync_path = sync_base / f"SWITCH{patient}_{phase}_sync"

    if not sync_path.exists():
        raise FileNotFoundError("bestand niet gevonden")

    sequence = load_sequence(sync_path)

    time_window, pes_window, t_start, t_end = get_pes_window(
        sequence=sequence,
        patient=patient,
        phase=phase,
        fs=fs,
    )

    eit_window = get_eit_window(
        sequence=sequence,
        t_start=t_start,
        t_end=t_end,
    )

    hr_eit_hz, HR_EIT_bpm = estimate_hr_eit(eit_window)

    (
        pes_bp, cgo, RCs, rc_pairs, chosen_pair, t_hr, hr, peaks, s,
        pair_scores, cgo_energy, pes_bp_energy, cgo_to_pesbp_ratio
    ) = process_pes(time_window, pes_window, fs)

    HR_CGO_bpm = np.nanmean(hr) if len(hr) > 0 else np.nan

    df_rc = rc_pair_overview_dataframe(
        pair_scores=pair_scores,
        RCs=RCs,
        rc_pairs=rc_pairs,
        fs=fs,
        hr_eit=hr_eit_hz,
    )

    df_rc["patient"] = patient
    df_rc["phase"] = phase
    df_rc["HR_CGO_bpm"] = HR_CGO_bpm
    df_rc["HR_EIT_bpm"] = HR_EIT_bpm
    df_rc["mismatch_rc_pair"] = df_rc["chosen_cgo"] != df_rc["chosen_hr_eit"]

    df_rc = add_bpm_columns(df_rc)

    return df_rc[WANTED_COLUMNS]


def collect_results(patients, phases, sync_base, fs, output_csv=None, decimals=2):
    all_dfs = []
    errors = []

    for patient in patients:
        for phase in phases:
            try:
                df_phase = analyse_patient_phase(patient, phase, sync_base, fs)
                all_dfs.append(df_phase)
                print(f"OK: patient {patient}, phase {phase}")

            except Exception as e:
                errors.append((patient, phase, str(e)))
                print(f"FOUT: patient {patient}, phase {phase} -> {e}")

    if not all_dfs:
        raise RuntimeError("Geen data verzameld. Controleer de inputbestanden en paden.")

    df_output = pd.concat(all_dfs, ignore_index=True)
    df_output = round_numeric_columns(df_output, decimals=decimals)

    if output_csv is not None:
        df_output.to_csv(output_csv, sep=";", index=False)
        print(f"\nCSV opgeslagen naar:\n{output_csv}")

    return df_output, errors


def apply_hr_eit_corrections(df, corrections, output_csv=None, decimals=2):
    df = df.copy()

    for (patient, phase), corrected_bpm in corrections.items():
        mask = (
            (df["patient"].astype(str).str.zfill(3) == patient) &
            (df["phase"] == phase)
        )

        if not mask.any():
            print(f"Waarschuwing: geen data gevonden voor patient {patient}, phase {phase}")
            continue

        df.loc[mask, "HR_EIT_bpm"] = corrected_bpm
        df.loc[mask, "f_target_bpm"] = corrected_bpm
        df.loc[mask, "chosen_hr_eit"] = False

        idx_best = df.loc[mask, "f_dom_bpm"].sub(corrected_bpm).abs().idxmin()
        df.loc[idx_best, "chosen_hr_eit"] = True

    df["mismatch_rc_pair"] = df["chosen_cgo"] != df["chosen_hr_eit"]
    df = round_numeric_columns(df, decimals=decimals)

    if output_csv is not None:
        df.to_csv(output_csv, sep=";", index=False)
        print(f"\nGecorrigeerde CSV opgeslagen naar:\n{output_csv}")

    return df