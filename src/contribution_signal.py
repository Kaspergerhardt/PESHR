from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append("../src")
from visualization import plot_contribution_distribution


def run_contribution_to_signal_analysis(csv_path, output_dir, tolerance_bpm=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, sep=";")

    df["freq_diff_bpm"] = (df["f_dom_bpm"] - df["HR_EIT_bpm"]).abs()
    df["eit_match"] = df["freq_diff_bpm"] <= tolerance_bpm

    chosen = df[df["chosen_cgo"]]

    correct_cases = chosen.loc[
        chosen["eit_match"], ["patient", "phase"]
    ].drop_duplicates()

    incorrect_cases = chosen.loc[
        ~chosen["eit_match"], ["patient", "phase"]
    ].drop_duplicates()

    group_a = chosen.merge(correct_cases, on=["patient", "phase"])
    group_a["group"] = "SSA selected RC pairs"

    candidates = df.merge(incorrect_cases, on=["patient", "phase"])

    group_b = (
        candidates[candidates["eit_match"]]
        .sort_values(["patient", "phase", "freq_diff_bpm"])
        .drop_duplicates(["patient", "phase"])
    )
    group_b["group"] = "EIT matched RC pairs"

    plot_contribution_distribution(
        group_a,
        group_b,
        save_path=output_dir / "contribution_distribution.png",
    )

    no_match_cases = incorrect_cases.merge(
        group_b[["patient", "phase"]],
        on=["patient", "phase"],
        how="left",
        indicator=True,
    ).query("_merge == 'left_only'")[["patient", "phase"]]

    rows = []

    for patient, phase in no_match_cases.itertuples(index=False):
        case_df = df[(df["patient"] == patient) & (df["phase"] == phase)]

        chosen_row = case_df[case_df["chosen_cgo"]]
        closest = case_df.loc[case_df["freq_diff_bpm"].idxmin()]

        rows.append({
            "Patient": patient,
            "Phase": phase,
            "HR CGO": chosen_row["HR_CGO_bpm"].iloc[0] if not chosen_row.empty else np.nan,
            "HR EIT": closest["HR_EIT_bpm"],
            "Closest freq.": closest["f_dom_bpm"],
            "Freq. diff.": closest["freq_diff_bpm"],
            "Contribution": closest["contribution"],
        })

    return pd.DataFrame(rows).round(2)