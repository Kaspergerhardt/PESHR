from pathlib import Path

import numpy as np
import pandas as pd
import sys

sys.path.append(r"../visualization")
from ssa_plots import (
    plot_clinical_impact_distribution,
    plot_clinical_impact_boxplot,
)


def split_clinical_impact_groups(df, tolerance_bpm=12, hr_diff_threshold=10):
    df = df.copy()

    df["desired_freq_bpm"] = df["HR_EIT_bpm"]
    df["freq_diff_bpm"] = (df["f_dom_bpm"] - df["desired_freq_bpm"]).abs()
    df["hr_diff_bpm"] = (df["HR_CGO_bpm"] - df["HR_EIT_bpm"]).abs()

    mismatch_cases = (
        df.loc[df["mismatch_rc_pair"], ["patient", "phase"]]
        .drop_duplicates()
    )

    group_a = (
        df.merge(mismatch_cases, on=["patient", "phase"], how="inner")
        .query("freq_diff_bpm <= @tolerance_bpm and hr_diff_bpm > @hr_diff_threshold")
        .copy()
        .sort_values(["patient", "phase", "freq_diff_bpm"])
    )

    matched_cases = group_a[["patient", "phase"]].drop_duplicates()

    group_b = (
        df.merge(
            matched_cases.assign(exclude=True),
            on=["patient", "phase"],
            how="left",
        )
        .query("exclude.isna() and chosen_cgo == True")
        .copy()
    )

    group_a["group"] = "Mismatch + EIT match"
    group_b["group"] = "Remaining"

    return group_a, group_b


def make_clinical_impact_summary(group_a, group_b):
    rows = []

    for name, group in [
        ("Mismatch + EIT match", group_a),
        ("Remaining", group_b),
    ]:
        values = group["clinical_impact"].dropna()

        rows.append({
            "group": name,
            "n": len(values),
            "mean": values.mean(),
            "median": values.median(),
            "min": values.min(),
            "max": values.max(),
            "std": values.std(),
        })

    summary = pd.DataFrame(rows)

    a = group_a["clinical_impact"].dropna()
    b = group_b["clinical_impact"].dropna()

    pooled_sd = np.sqrt((a.var() + b.var()) / 2)
    cohen_d = (a.mean() - b.mean()) / pooled_sd

    summary["cohen_d_A_vs_B"] = cohen_d

    return summary.round(2)


def save_clinical_impact_overview(group_a, group_b, save_path):
    columns = [
        "group",
        "patient",
        "phase",
        "pair",
        "clinical_impact",
        "HR_CGO_bpm",
        "HR_EIT_bpm",
        "hr_diff_bpm",
        "f_dom_bpm",
        "desired_freq_bpm",
        "freq_diff_bpm",
    ]

    overview = pd.concat([group_a, group_b], ignore_index=True)

    available_columns = [col for col in columns if col in overview.columns]
    overview = overview[available_columns]

    overview.to_csv(save_path, sep=";", index=False)

    return overview


def run_clinical_impact_analysis(
    csv_path,
    output_dir,
    tolerance_bpm=12,
    hr_diff_threshold=10,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, sep=";")

    group_a, group_b = split_clinical_impact_groups(
        df,
        tolerance_bpm=tolerance_bpm,
        hr_diff_threshold=hr_diff_threshold,
    )

    overview = save_clinical_impact_overview(
        group_a,
        group_b,
        output_dir / "clinical_impact_overview.csv",
    )

    summary = make_clinical_impact_summary(group_a, group_b)
    summary.to_csv(
        output_dir / "clinical_impact_summary.csv",
        sep=";",
        index=False,
    )

    plot_clinical_impact_distribution(
        group_a,
        group_b,
        save_path=output_dir / "clinical_impact_distribution.png",
    )

    plot_clinical_impact_boxplot(
        group_a,
        group_b,
        save_path=output_dir / "clinical_impact_boxplot.png",
    )

    return overview, summary, group_a, group_b