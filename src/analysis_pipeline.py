import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("..").resolve() / "eitprocessing"))
sys.path.append("../filter")
sys.path.append("../config")
sys.path.append("../src")

from eitprocessing.features.rate_detection import RateDetection
from SSA_HR import process_pes, rc_pair_overview_dataframe
from dataset_config import get_dataset_window
from loading import (
    load_synchronized_sequence,
    load_synchronized_pes,
    load_synchronized_eit,
    crop_pes_to_window,
    crop_eit_to_window,
)


WANTED_COLUMNS = [
    "patient", "phase", "pair",
    "cardiac_energy", "respiratory_energy", "score", "contribution",
    "f_dom_bpm", "f_target_bpm",
    "chosen_cgo", "chosen_hr_eit", "mismatch_rc_pair",
    "HR_CGO_bpm", "HR_EIT_bpm",
]


def round_numeric_columns(df, decimals=2):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].round(decimals)
    return df


def add_bpm_columns(df):
    df = df.copy()

    if "f_dom_hz" in df.columns:
        df["f_dom_bpm"] = df["f_dom_hz"] * 60

    if "f_target_hz" in df.columns:
        df["f_target_bpm"] = df["f_target_hz"] * 60

    return df


def get_pes_window(sequence, patient, phase):
    time, pes = load_synchronized_pes(sequence)

    t_start, t_end = get_dataset_window(patient, phase)
    t_start += time[0]
    t_end += time[0]

    time_window, pes_window = crop_pes_to_window(
        time=time,
        pes=pes,
        t_start=t_start,
        t_end=t_end,
    )

    return time_window, pes_window, t_start, t_end


def get_eit_window(sequence, t_start, t_end):
    eit_signal, _ = load_synchronized_eit(sequence)

    eit_window, _, _ = crop_eit_to_window(
        eit_signal=eit_signal,
        t_start=t_start,
        t_end=t_end,
    )

    return eit_window


def estimate_hr_eit(eit_window):
    rd = RateDetection(subject_type="adult")
    _, hr_eit_hz = rd.apply(eit_window, captures={})

    hr_eit_bpm = hr_eit_hz * 60 if pd.notna(hr_eit_hz) else np.nan
    return hr_eit_hz, hr_eit_bpm


def analyse_patient_phase(patient, phase, sync_base, fs):
    sync_path = Path(sync_base) / f"SWITCH{patient}_{phase}_sync"
    sequence = load_synchronized_sequence(sync_path)

    time_window, pes_window, t_start, t_end = get_pes_window(
        sequence=sequence,
        patient=patient,
        phase=phase,
    )

    eit_window = get_eit_window(sequence, t_start, t_end)
    hr_eit_hz, hr_eit_bpm = estimate_hr_eit(eit_window)

    (
        pes_bp, cgo, RCs, rc_pairs, chosen_pair, t_hr, hr, peaks, s,
        pair_scores, cgo_energy, pes_bp_energy, cgo_to_pesbp_ratio
    ) = process_pes(time_window, pes_window, fs)

    hr_cgo_bpm = np.nanmean(hr) if len(hr) else np.nan

    df_rc = rc_pair_overview_dataframe(
        pair_scores=pair_scores,
        RCs=RCs,
        rc_pairs=rc_pairs,
        fs=fs,
        hr_eit=hr_eit_hz,
    )

    df_rc = add_bpm_columns(df_rc)

    df_rc = df_rc.assign(
        patient=patient,
        phase=phase,
        HR_CGO_bpm=hr_cgo_bpm,
        HR_EIT_bpm=hr_eit_bpm,
        mismatch_rc_pair=lambda x: x["chosen_cgo"] != x["chosen_hr_eit"],
    )

    return df_rc[WANTED_COLUMNS]


def collect_results(patients, phases, sync_base, fs, output_csv=None, decimals=2):
    results = []
    errors = []

    for patient in patients:
        for phase in phases:
            try:
                df_phase = analyse_patient_phase(patient, phase, sync_base, fs)
                results.append(df_phase)
                print(f"OK: patient {patient}, phase {phase}")

            except Exception as e:
                errors.append((patient, phase, str(e)))
                print(f"FOUT: patient {patient}, phase {phase} -> {e}")

    if not results:
        raise RuntimeError("Geen data verzameld. Controleer de inputbestanden en paden.")

    df_output = pd.concat(results, ignore_index=True)
    df_output = round_numeric_columns(df_output, decimals)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        df_output.to_csv(output_csv, sep=";", index=False)
        print(f"\nCSV opgeslagen naar:\n{output_csv}")

    return df_output, errors


def apply_hr_eit_corrections(df, corrections, output_csv=None, decimals=2):
    df = df.copy()
    patient_ids = df["patient"].astype(str).str.zfill(3)

    for (patient, phase), corrected_bpm in corrections.items():
        mask = (
            (patient_ids == str(patient).zfill(3)) &
            (df["phase"] == phase)
        )

        if not mask.any():
            print(f"Waarschuwing: geen data gevonden voor patient {patient}, phase {phase}")
            continue

        df.loc[mask, ["HR_EIT_bpm", "f_target_bpm"]] = corrected_bpm
        df.loc[mask, "chosen_hr_eit"] = False

        best_idx = df.loc[mask, "f_dom_bpm"].sub(corrected_bpm).abs().idxmin()
        df.loc[best_idx, "chosen_hr_eit"] = True

    df["mismatch_rc_pair"] = df["chosen_cgo"] != df["chosen_hr_eit"]
    df = round_numeric_columns(df, decimals)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_csv, sep=";", index=False)
        print(f"\nGecorrigeerde CSV opgeslagen naar:\n{output_csv}")

    return df