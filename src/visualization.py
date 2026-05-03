import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_pes_segment(
    time,
    pes_filt,
    time_pes_plot,
    pes_plot,
    t_start,
    t_end,
    patient,
    phase,
):
    # Full PES signal
    plt.figure(figsize=(12, 4))
    plt.plot(time, pes_filt, label="PES")
    plt.axvline(t_start, color="r", linestyle="--", label="segment start")
    plt.axvline(t_end, color="r", linestyle="--", label="segment eind")
    plt.xlabel("Tijd (s)")
    plt.ylabel("Pes (cmH2O)")
    plt.title(f"SWITCH{patient} - {phase}")
    plt.grid(True)
    plt.legend()
    plt.show()

    # PES window
    plt.figure(figsize=(12, 4))
    plt.plot(time_pes_plot, pes_plot, label="PES (window)")
    plt.axvline(t_start, color="r", linestyle="--", label="segment start")
    plt.axvline(t_end, color="r", linestyle="--", label="segment eind")
    plt.xlabel("Tijd (s)")
    plt.ylabel("Pes (cmH2O)")
    plt.title(f"SWITCH{patient} - {phase} - venster")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_ssa_overview(s, RCs, rc_pairs, pair_scores, fs, chosen_pair, pair_to_highlight=None, n_values=10):
    if pair_to_highlight is None:
        pair_to_highlight = chosen_pair

    # ------------------------------ 
    #  1. Eigenvalues en indexen
    # ------------------------------
    eigenvalues = s**2
    n_plot = min(n_values, len(eigenvalues))
    idx = np.arange(1, n_plot + 1)

    plt.figure(figsize=(10, 5))
    plt.semilogy(idx, eigenvalues[:n_plot], marker='o', linewidth=1)
    plt.title("SSA Eigenvalues")
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---------------------
    # 2. RC's + RC-paren 
    # ---------------------
    fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for i in range(min(10, RCs.shape[0])):
        x = RCs[i] - np.mean(RCs[i])
        freqs = np.fft.rfftfreq(len(x), d=1/fs)
        psd = np.abs(np.fft.rfft(x))**2
        mask = (freqs >= 0.4) & (freqs <= 4.0)
        axs[0].plot(freqs[mask], psd[mask], label=f"RC {i}")

    axs[0].set_title("Spectral representation of RCs")
    axs[0].set_ylabel("Power")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=8, ncol=2)

    for pair in rc_pairs:
        rc_pair_signal = np.sum(RCs[list(pair)], axis=0)
        x = rc_pair_signal - np.mean(rc_pair_signal)
        freqs = np.fft.rfftfreq(len(x), d=1/fs)
        psd = np.abs(np.fft.rfft(x))**2
        mask = (freqs >= 0.4) & (freqs <= 4.0)

        if pair == pair_to_highlight:
            axs[1].plot(
                freqs[mask], psd[mask],
                linewidth=3,
                label=f"{pair}"
            )
        else:
            axs[1].plot(
                freqs[mask], psd[mask],
                alpha=0.6,
                linewidth=1.5,
                label=f"{pair}"
            )

    axs[1].set_title("Spectral representation of RC pairs")
    axs[1].set_xlabel("Frequency [bpm]")
    axs[1].set_ylabel("Power")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()



def plot_cgo_analysis(
    time_window,
    pes_window,
    pes_bp,
    cgo,
    peaks,
    t_hr,
    hr,
):
    # ----------------------------
    # Plot 
    # ----------------------------
    t_start = time_window[0]
    t_end = time_window[-1]

    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=False)
    fig.suptitle("CGO analysis", fontsize=14)

    # -----------------------------------------------
    # Helper for dynamic y-limits
    # -----------------------------------------------
    def dynamic_ylim(signal, margin=0.1):
        signal = np.asarray(signal)

        if len(signal) == 0:
            return -1, 1

        y_min = np.min(signal)
        y_max = np.max(signal)
        y_range = y_max - y_min

        if y_range == 0:
            y_range = max(abs(y_min), 1.0)

        return y_min - margin * y_range, y_max + margin * y_range

    # ---------------------------
    # 1. Original PES signal
    # ---------------------------
    axes[0].plot(time_window, pes_window, label="Original Pes", alpha=0.7)
    axes[0].set_title("Original Pes")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(t_start, t_end)
    axes[0].set_ylim(*dynamic_ylim(pes_window))
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # ---------------------------
    # 2. Bandgefilterd PES
    # ---------------------------
    axes[1].plot(time_window, pes_bp, label="Bandpassed Pes", alpha=0.7)
    axes[1].set_title("Bandpassed Pes")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlim(t_start, t_end)
    axes[1].set_ylim(*dynamic_ylim(pes_bp))
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # ------------------------
    # 3. Extracted CGO + peaks
    # ------------------------
    axes[2].plot(time_window, cgo, label="Extracted CGO", linewidth=2)

    if len(peaks) > 0:
        axes[2].plot(
            time_window[peaks],
            cgo[peaks],
            "o",
            label="Detected peaks",
            markersize=5
        )

        for p_i in peaks:
            axes[2].axvline(time_window[p_i], linestyle="--", alpha=0.2)

    axes[2].set_title("CGO + peak detection")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlim(t_start, t_end)
    axes[2].set_ylim(*dynamic_ylim(cgo))
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    # -------------------------------------------
    # 4. Origineel PES + CGO pieken (verticale lijnen)
    # -------------------------------------------
    axes[3].plot(time_window, pes_window, label="Original Pes", alpha=0.7)

    if len(peaks) > 0:
        for p_i in peaks:
            axes[3].axvline(
                time_window[p_i],
                linestyle="--",
                alpha=0.3,
                color="r"
            )

    axes[3].set_title("Original Pes with detected peaks")
    axes[3].set_ylabel("Amplitude")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_xlim(t_start, t_end)
    axes[3].set_ylim(*dynamic_ylim(pes_window))
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    # -------------
    # 5. Heart rate
    # -------------
    if len(hr) > 0:
        axes[4].plot(t_hr, hr, label="Heart rate", linewidth=2)

    axes[4].set_title("Heart rate")
    axes[4].set_ylabel("BPM")
    axes[4].set_xlabel("Time [s]")
    axes[4].set_xlim(t_start, t_end)

    if len(hr) > 0:
        axes[4].set_ylim(*dynamic_ylim(hr))

    axes[4].legend(loc="upper right")
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    # ---------------
    # Statistics
    # ---------------
    if len(hr) > 0:
        print(f"Gemiddelde hartslag uit pieken: {np.mean(hr):.2f} bpm")
        print(f"Mediaan hartslag uit pieken:    {np.median(hr):.2f} bpm")
    else:
        print("Geen geldige HR bepaald uit piekdetectie.")


def bland_altman_plot(
    csv_path,
    save_path=None,
    col_ref="HR_EIT_bpm",
    col_test="HR_CGO_bpm",
    sep=";",
    title=None,
    filter_condition=None,
    show_plot=True
):
    """
    Create a Bland-Altman plot with ±5 bpm agreement analysis.
    """

    # -----------------------
    # Load data
    # -----------------------
    df = pd.read_csv(csv_path, sep=sep)

    # Optional filtering
    if filter_condition is not None:
        df = df[filter_condition(df)]

    # Keep only relevant columns
    df = df[[col_ref, col_test]].dropna()

    if len(df) == 0:
        print("No data available after filtering.")
        return

    # -----------------------
    # Bland-Altman metrics
    # -----------------------
    mean_vals = (df[col_ref] + df[col_test]) / 2
    diff_vals = df[col_test] - df[col_ref]
    

    bias = diff_vals.mean()
    sd = diff_vals.std(ddof=1)
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    # -----------------------
    # ±5 bpm analyse
    # -----------------------
    outside_5 = np.sum(np.abs(diff_vals) > 5)
    total = len(diff_vals)
    percentage_outside_5 = (outside_5 / total) * 100

    # -----------------------
    # Plot
    # -----------------------
    plt.figure(figsize=(8, 6))

    mask_outside = np.abs(diff_vals) > 5
    mask_inside = ~mask_outside

    plt.scatter(mean_vals[mask_inside], diff_vals[mask_inside], alpha=0.8, label="Binnen ±5 bpm")
    plt.scatter(mean_vals[mask_outside], diff_vals[mask_outside], label="Buiten ±5 bpm")

    plt.axhline(bias, linestyle="--", label=f"Bias = {bias:.2f}")
    plt.axhline(loa_upper, linestyle=":", label=f"+1.96 SD = {loa_upper:.2f}")
    plt.axhline(loa_lower, linestyle=":", label=f"-1.96 SD = {loa_lower:.2f}")
    plt.axhline(0, linestyle="-", label="0 bpm")
    plt.axhline(5, linestyle="-.", label="+5 bpm")
    plt.axhline(-5, linestyle="-.", label="-5 bpm")

    plt.xlabel(f"Mean ({col_ref} & {col_test}) [bpm]")
    plt.ylabel(f"Difference ({col_test} - {col_ref}) [bpm]")

    if title is None:
        title = f"Bland-Altman: {col_test} vs {col_ref}"

    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()

    # -----------------------
    # Output stats
    # -----------------------
    print(f"N = {total}")
    print(f"Bias: {bias:.2f} bpm")
    print(f"95% LoA: [{loa_lower:.2f}, {loa_upper:.2f}] bpm")
    print(f"Aantal buiten ±5 bpm: {outside_5}")
    print(f"Percentage buiten ±5 bpm: {percentage_outside_5:.1f}%")


def plot_contribution_distribution(group_a, group_b, save_path=None):
    plt.figure(figsize=(8, 5))

    label_a = group_a["group"].iloc[0]
    label_b = group_b["group"].iloc[0]

    plt.hist(
        group_a["contribution"].dropna(),
        bins=15,
        alpha=0.6,
        density=True,
        label=label_a,
    )

    plt.hist(
        group_b["contribution"].dropna(),
        bins=15,
        alpha=0.6,
        density=True,
        label=label_b,
    )

    plt.xlabel("Contribution")
    plt.ylabel("Density")
    plt.title("Distribution of signal contribution")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


