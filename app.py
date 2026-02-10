import pathlib
import streamlit as st
import numpy as np
import scipy.signal as sig
import librosa
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.integrate
from scipy.ndimage import gaussian_filter1d

APP_DIR = pathlib.Path(__file__).parent

# ---------------------------------------------------------------------------
# Page config & theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Acoustic Uroflowmetry Analyzer",
    page_icon=":material/water_drop:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for metric cards and layout polish
st.markdown("""<style>
    [data-testid="stMetric"] {
        border-left: 4px solid #0066CC;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .block-container {
        padding-top: 1.5rem;
    }
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Matplotlib clinical style — applied globally
# ---------------------------------------------------------------------------
CLINICAL_STYLE = {
    'figure.facecolor': 'white',
    'figure.dpi': 100,
    'axes.facecolor': 'white',
    'axes.edgecolor': '#CCCCCC',
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'axes.axisbelow': True,
    'axes.titlesize': 14,
    'axes.titleweight': 600,
    'axes.titlepad': 12,
    'axes.labelsize': 11,
    'axes.labelcolor': '#333333',
    'axes.labelpad': 8,
    'axes.prop_cycle': mpl.cycler(color=[
        '#0066CC', '#E63946', '#2A9D8F', '#E9C46A', '#6C757D',
    ]),
    'grid.color': '#E8E8E8',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.7,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.color': '#555555',
    'ytick.color': '#555555',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'lines.linewidth': 1.8,
    'lines.antialiased': True,
    'legend.fontsize': 9,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#CCCCCC',
    'legend.fancybox': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
}
plt.rcParams.update(CLINICAL_STYLE)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    logo_path = APP_DIR / "UrologyMYLogo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=180)
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload Audio (WAV)", type=["wav"])
    total_volume_ml = st.number_input(
        "Total Voided Volume (mL)", min_value=1, value=None, step=1, format="%d",
    )
    is_ready = uploaded_file is not None and total_volume_ml is not None
    start_button = st.button(
        "Start Processing", disabled=not is_ready,
        type="primary", use_container_width=True,
    )
    st.divider()
    st.caption(
        "**Dr Badrulhisham Bahadzor**  \n"
        "Consultant Urologist  \n"
        "[drbadrul@urology.my](mailto:drbadrul@urology.my)  \n"
        "Visit [www.urology.my](https://www.urology.my) to know more."
    )

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("Acoustic Uroflowmetry Analysis")
st.caption("Audio-based flow curve signal analysis")

# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------

def process_audio(file, volume_ml):
    # Collection dicts for diagnostic info displayed later in expanders
    diag = {}
    qmax_warning = None

    # ==================================================================
    # COMPUTATION — wrapped in st.status for live progress feedback
    # ==================================================================
    with st.status("Analysing audio signal...", expanded=True) as status:

        # ----------------------------------------------------------
        # Step 1: Signal Conditioning & RMS
        # ----------------------------------------------------------
        status.write("Step 1 — Loading and conditioning signal...")

        y, sr = librosa.load(file, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        sos = sig.butter(4, [300, 2500], btype='bandpass', fs=sr, output='sos')
        y_filtered = sig.sosfiltfilt(sos, y)

        frame_length = 1024
        hop_length = 512
        rms = librosa.feature.rms(
            y=y_filtered, frame_length=frame_length,
            hop_length=hop_length, center=True,
        )[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        diag['duration'] = duration
        diag['sr'] = sr

        # --- fig1: raw RMS ---
        fig1, ax1 = plt.subplots(figsize=(10, 3.5))
        ax1.plot(times, rms)
        ax1.set_title("Raw RMS Amplitude vs. Time")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        plt.tight_layout()

        # ----------------------------------------------------------
        # Step 2: Pre-Processing (Artifact Removal & Smoothing)
        # ----------------------------------------------------------
        status.write("Step 2 — Artifact removal and pre-smoothing...")

        cent = librosa.feature.spectral_centroid(
            y=y_filtered, sr=sr, n_fft=frame_length, hop_length=hop_length,
        )[0]

        # Otsu threshold
        rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-9)
        try:
            from skimage.filters import threshold_otsu
            otsu_val_norm = threshold_otsu(rms_norm)
            otsu_thresh = otsu_val_norm * (np.max(rms) - np.min(rms)) + np.min(rms)
        except ImportError:
            counts, bin_edges = np.histogram(rms, bins=256)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            weight_total = np.sum(counts)
            sum_total = np.sum(bin_centers * counts)
            best_thresh = 0
            max_variance = 0
            weight_bg = 0
            sum_bg = 0
            for i in range(len(counts)):
                weight_bg += counts[i]
                if weight_bg == 0: continue
                weight_fg = weight_total - weight_bg
                if weight_fg == 0: break
                sum_bg += bin_centers[i] * counts[i]
                mean_bg = sum_bg / weight_bg
                mean_fg = (sum_total - sum_bg) / weight_fg
                between_variance = weight_bg * weight_fg * ((mean_bg - mean_fg) ** 2)
                if between_variance > max_variance:
                    max_variance = between_variance
                    best_thresh = bin_centers[i]
            otsu_thresh = best_thresh

        # Artifact removal — read neighbours from ORIGINAL rms
        rms_clean = rms.copy()
        artifact_indices = np.where((rms > otsu_thresh) & (cent < 250))[0]
        for i in artifact_indices:
            start_k = max(0, i - 5)
            end_k = min(len(rms), i + 6)
            neighbours = np.concatenate([rms[start_k:i], rms[i + 1:end_k]])
            if len(neighbours) > 0:
                rms_clean[i] = np.min(neighbours)

        diag['n_artifacts'] = len(artifact_indices)

        # Pre-event Savitzky-Golay smoothing (stabilises dribble for detection)
        savgol_window = 51
        if len(rms_clean) < savgol_window:
            savgol_window = len(rms_clean)
            if savgol_window % 2 == 0:
                savgol_window -= 1
        if savgol_window > 3:
            rms_smoothed_full = sig.savgol_filter(rms_clean, window_length=savgol_window, polyorder=3)
        else:
            rms_smoothed_full = rms_clean.copy()
        rms_smoothed_full = np.maximum(rms_smoothed_full, 0)

        # ----------------------------------------------------------
        # Step 3: Event Detection (Hysteresis + Adaptive Hangover)
        # ----------------------------------------------------------
        status.write("Step 3 — Detecting flow events...")

        rms_detect = rms_smoothed_full

        # 1. Otsu trigger
        trigger_indices = np.where(rms_detect > otsu_thresh)[0]
        if len(trigger_indices) == 0:
            st.error("No significant flow detected (Otsu trigger failed).")
            return
        trigger_idx = trigger_indices[0]

        # 2. Stable noise floor
        frames_200ms = max(1, int(0.2 * sr / hop_length))
        if trigger_idx < frames_200ms:
            best_win_start = 0
            best_win_end = min(len(rms_detect), frames_200ms)
        else:
            search_limit = trigger_idx
            min_std = float('inf')
            best_win_start = 0
            for start_i in range(0, search_limit - frames_200ms + 1):
                end_i = start_i + frames_200ms
                curr_std = np.std(rms_detect[start_i:end_i])
                if curr_std < min_std:
                    min_std = curr_std
                    best_win_start = start_i
            best_win_end = best_win_start + frames_200ms

        noise_window = rms_detect[best_win_start:best_win_end]
        noise_mean = np.mean(noise_window)
        noise_std = np.std(noise_window)

        onset_threshold = noise_mean + 5 * noise_std
        offset_threshold = noise_mean + 2 * noise_std

        diag['noise_win'] = (times[best_win_start], times[best_win_end])
        diag['noise_mean'] = noise_mean
        diag['noise_std'] = noise_std
        diag['onset_threshold'] = onset_threshold
        diag['offset_threshold'] = offset_threshold

        # 3. Onset
        onset_idx = -1
        for i in range(best_win_end, len(rms_detect)):
            if rms_detect[i] > onset_threshold:
                onset_idx = i
                break
        if onset_idx == -1:
            st.error("Could not detect onset based on calibrated threshold.")
            return

        # 4. Peak
        peak_relative_idx = np.argmax(rms_detect[onset_idx:])
        peak_idx = onset_idx + peak_relative_idx

        # 5. Adaptive hangover
        post_peak_rms = rms_detect[peak_idx:]
        post_peak_times = times[peak_idx:] - times[peak_idx]
        peak_val = rms_detect[peak_idx]
        decay_target = peak_val / np.e

        decay_indices = np.where(post_peak_rms < decay_target)[0]
        if len(decay_indices) > 0 and post_peak_times[decay_indices[0]] > 0:
            decay_time = post_peak_times[decay_indices[0]]
            hangover_duration = float(np.clip(decay_time * 2.0, 2.0, 8.0))
        else:
            hangover_duration = 4.0

        frames_hangover = int(hangover_duration * sr / hop_length)
        diag['hangover'] = hangover_duration

        # 6. Hysteresis offset scan
        offset_idx = len(rms_detect) - 1
        last_above_offset_idx = peak_idx
        timer = 0
        for i in range(peak_idx, len(rms_detect)):
            if rms_detect[i] >= offset_threshold:
                last_above_offset_idx = i
                timer = 0
            else:
                timer += 1
                if timer >= frames_hangover:
                    offset_idx = last_above_offset_idx
                    break

        # 7. Spectral-flux validation
        onset_env = librosa.onset.onset_strength(y=y_filtered, sr=sr, hop_length=hop_length)
        flux_noise_region = onset_env[best_win_start:best_win_end]
        flux_baseline = np.mean(flux_noise_region)
        flux_noise_std = np.std(flux_noise_region)
        flux_threshold = flux_baseline + 3 * flux_noise_std

        check_window = max(1, int(0.5 * sr / hop_length))
        search_end = min(len(rms_detect), len(onset_env), offset_idx + int(10.0 * sr / hop_length))
        extended_offset = offset_idx
        i = offset_idx + 1
        while i + check_window <= search_end:
            rms_win = rms_detect[i:i + check_window]
            flux_win = onset_env[i:i + check_window]
            if np.mean(rms_win) > noise_mean + 1.5 * noise_std and np.sum(flux_win > flux_threshold) >= 2:
                extended_offset = i + check_window
                i += check_window
            else:
                break

        diag['spectral_ext'] = 0.0
        if extended_offset > offset_idx:
            diag['spectral_ext'] = times[extended_offset] - times[offset_idx]
            offset_idx = extended_offset

        # 8. Changepoint detection (PELT)
        diag['cp_ext'] = 0.0
        try:
            import ruptures
            cp_buffer = int(5.0 * sr / hop_length)
            cp_end = min(len(rms_detect), offset_idx + cp_buffer)
            cp_signal = rms_detect[peak_idx:cp_end].reshape(-1, 1)
            if len(cp_signal) > 10:
                min_size = max(int(0.5 * sr / hop_length), 2)
                algo = ruptures.Pelt(model="l2", min_size=min_size).fit(cp_signal)
                penalty = float(np.std(cp_signal) ** 2 * np.log(len(cp_signal)))
                changepoints = algo.predict(pen=penalty)
                if len(changepoints) > 1:
                    last_cp_absolute = peak_idx + changepoints[-2]
                    if last_cp_absolute > offset_idx:
                        segment = rms_detect[offset_idx:last_cp_absolute]
                        if np.mean(segment) > noise_mean + 1.5 * noise_std:
                            diag['cp_ext'] = times[last_cp_absolute] - times[offset_idx]
                            offset_idx = last_cp_absolute
        except ImportError:
            pass
        except Exception:
            pass

        # Extend onset backwards to where signal meets the noise floor.
        # onset_idx is the high-confidence detection point (5σ), but the
        # actual start of flow is earlier — where the signal rises above
        # offset_threshold (2σ).  After zero-floor subtraction (which
        # subtracts offset_threshold), the extended region naturally starts
        # at zero, giving a physiologically realistic ramp-up.
        extended_onset = onset_idx
        for i in range(onset_idx - 1, max(best_win_end - 1, -1), -1):
            if rms_detect[i] <= offset_threshold:
                extended_onset = i
                break
        else:
            # Signal never dropped below offset_threshold — use noise window end
            extended_onset = best_win_end

        final_onset = extended_onset
        final_offset = min(offset_idx, len(rms_detect) - 1)

        # Trim using CLEAN signal (original energy)
        rms_trimmed = rms_clean[final_onset : final_offset + 1]
        times_trimmed = times[final_onset : final_offset + 1] - times[final_onset]

        # --- fig2: event detection ---
        plot_start_idx = max(0, best_win_start - int(1.0 * sr / hop_length))
        plot_end_idx = min(len(rms_detect), final_offset + int(2.0 * sr / hop_length))
        plot_rms = rms_detect[plot_start_idx : plot_end_idx]
        plot_times = times[plot_start_idx : plot_end_idx] - times[final_onset]

        fig2, ax2 = plt.subplots(figsize=(10, 3.5))
        ax2.plot(plot_times, plot_rms, label='Pre-smoothed Signal')
        n_s = times[best_win_start] - times[final_onset]
        n_e = times[best_win_end] - times[final_onset]
        ax2.axvspan(n_s, n_e, color='gray', alpha=0.3, label='Noise Window')
        ax2.axvline(0.0, color='#2A9D8F', linestyle='--', label='Onset')
        ax2.axvline(times[final_offset] - times[final_onset], color='#E63946', linestyle='--', label='Offset')
        ax2.axhline(onset_threshold, color='#E9C46A', linestyle=':', label=f'Onset Thr (5\u03c3)')
        ax2.axhline(offset_threshold, color='#0066CC', linestyle=':', label=f'Offset Thr (2\u03c3)')
        ax2.axhspan(offset_threshold, onset_threshold, color='#E9C46A', alpha=0.08)
        ax2.set_title("Hysteresis Event Detection")
        ax2.set_xlabel("Time (s) [Onset = 0]")
        ax2.set_ylabel("Amplitude")
        ax2.legend(fontsize=8, ncol=3, loc='upper right')
        plt.tight_layout()

        # ----------------------------------------------------------
        # Step 4: Zero-Floor Subtraction & Gaussian Smoothing
        # ----------------------------------------------------------
        status.write("Step 4 — Zero-floor subtraction and Gaussian smoothing...")

        rms_floored = np.maximum(rms_trimmed - offset_threshold, 0)

        # Force near-zero residuals to exactly zero.
        # After subtracting offset_threshold (2σ), values below 1σ are
        # still within noise variability — not real flow.  Zeroing them
        # creates clean pause boundaries for segment-wise smoothing.
        rms_floored[rms_floored < noise_std] = 0

        # Segment-wise Gaussian smoothing.
        # Zero pauses indicate intermittent flow and must be preserved.
        # We smooth each contiguous non-zero segment independently so
        # noise is removed within flow bursts while pauses stay at zero.
        gauss_sigma = max(1, int(0.015 * len(rms_floored)))
        rms_final = np.zeros_like(rms_floored)
        nonzero_mask = rms_floored > 0
        if np.any(nonzero_mask):
            # Find contiguous non-zero runs
            diff = np.diff(nonzero_mask.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1
            # Handle edge cases: signal starts or ends non-zero
            if nonzero_mask[0]:
                starts = np.concatenate([[0], starts])
            if nonzero_mask[-1]:
                ends = np.concatenate([ends, [len(rms_floored)]])
            for s, e in zip(starts, ends):
                segment = rms_floored[s:e]
                seg_sigma = max(1, int(0.015 * len(segment)))
                smoothed = gaussian_filter1d(segment, sigma=seg_sigma, mode='constant', cval=0.0)
                rms_final[s:e] = np.maximum(smoothed, 0)

        # Adaptive half-cosine onset ramp applied from frame 0.
        # When noise_std is very small, the backward onset extension
        # gains only 1-2 frames, causing a near-vertical jump at t=0.
        # The ramp covers the first N frames unconditionally — frames
        # that are already 0 stay at 0 (0 × window = 0), so leading
        # zeros and any micro-gaps within the ramp region are preserved.
        # By the time a real intermittent pause occurs (seconds in),
        # the window value is already 1.0, so no attenuation happens.
        peak_trimmed_idx = peak_idx - final_onset
        time_to_peak = times_trimmed[min(peak_trimmed_idx, len(times_trimmed) - 1)]
        ramp_duration_sec = np.clip(0.10 * time_to_peak, 0.5, 2.0)

        nz_mask = rms_final > 0
        ramp_applied = False
        if np.any(nz_mask):
            dt = times_trimmed[1] - times_trimmed[0] if len(times_trimmed) > 1 else 1.0
            ramp_frames = min(int(ramp_duration_sec / dt), len(rms_final))
            if ramp_frames > 1:
                t_ramp = np.linspace(0, np.pi, ramp_frames, endpoint=True)
                window = 0.5 * (1 - np.cos(t_ramp))
                rms_final[:ramp_frames] *= window
                ramp_applied = True

        diag['ramp_duration'] = ramp_duration_sec

        # Trim trailing low-amplitude tail (post-flow noise).
        # The offset detection can overshoot because the signal decays
        # gradually through the threshold without a clean zero gap.
        # We define "effective end of flow" as the last point where
        # amplitude >= 2 % of peak, then continue to the next zero
        # crossing for a clean cutoff.
        peak_amp = np.max(rms_final) if len(rms_final) > 0 else 0
        if peak_amp > 0:
            tail_threshold = 0.02 * peak_amp
            above = np.where(rms_final >= tail_threshold)[0]
            if len(above) > 0:
                last_significant = above[-1]
                # Extend to next zero for a natural decay endpoint
                trim_idx = last_significant + 1
                while trim_idx < len(rms_final) and rms_final[trim_idx] > 0:
                    trim_idx += 1
                rms_final = rms_final[:trim_idx]
                times_trimmed = times_trimmed[:trim_idx]

        diag['gauss_sigma'] = gauss_sigma

        # --- fig3: processed flow ---
        fig3, ax3 = plt.subplots(figsize=(10, 3.5))
        ax3.plot(times_trimmed, rms_final, color='#2A9D8F')
        ax3.fill_between(times_trimmed, rms_final, alpha=0.15, color='#2A9D8F')
        ax3.set_title("Zero-Floored & Gaussian-Smoothed Flow")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Amplitude")
        ax3.set_ylim(bottom=0)
        plt.tight_layout()

        # ----------------------------------------------------------
        # Step 5: Normalisation & Flow Rate
        # ----------------------------------------------------------
        status.write("Step 5 — Normalising to flow rate...")

        auc = scipy.integrate.trapezoid(rms_final, times_trimmed)
        if auc == 0:
            st.error("AUC is zero. Cannot normalise.")
            return

        K = volume_ml / auc
        flow_rate = rms_final * K

        # Qmax validation (sustained 90 % of peak for >= 200 ms)
        frames_200ms = max(1, int(0.2 * sr / hop_length))
        peaks, _ = sig.find_peaks(flow_rate)
        valid_qmax_idx = -1
        valid_qmax_val = 0

        if len(peaks) > 0:
            for p in sorted(peaks, key=lambda p: flow_rate[p], reverse=True):
                val = flow_rate[p]
                above_ids = np.where(flow_rate >= 0.9 * val)[0]
                for group in np.split(above_ids, np.where(np.diff(above_ids) != 1)[0] + 1):
                    if p in group:
                        if len(group) >= frames_200ms:
                            valid_qmax_idx = p
                            valid_qmax_val = val
                        break
                if valid_qmax_idx != -1:
                    break

        if valid_qmax_idx == -1:
            valid_qmax_idx = int(np.argmax(flow_rate))
            valid_qmax_val = float(flow_rate[valid_qmax_idx])
            qmax_warning = "No peak met the 200 ms sustain criteria. Using absolute max."

        qmax = valid_qmax_val
        voiding_time = times_trimmed[-1] - times_trimmed[0] if len(times_trimmed) > 1 else 0
        average_flow = volume_ml / voiding_time if voiding_time > 0 else 0

        # --- fig4: polished clinical chart ---
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.fill_between(times_trimmed, flow_rate, alpha=0.12, color='#0066CC')
        ax4.plot(times_trimmed, flow_rate, color='#0066CC', linewidth=2.2, label='Flow Rate')
        ax4.axhline(qmax, color='#E63946', linestyle='--', linewidth=1.4, label=f'Qmax: {qmax:.1f} mL/s')
        ax4.axhline(average_flow, color='#2A9D8F', linestyle=':', linewidth=1.2, label=f'Avg: {average_flow:.1f} mL/s')
        ax4.set_title("Uroflowmetry Curve", fontsize=16, fontweight=600)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Flow Rate (mL/s)")
        ax4.set_ylim(bottom=0)
        ax4.legend(loc='upper right')
        plt.tight_layout()

        status.update(label="Analysis complete!", state="complete", expanded=False)

    # ==================================================================
    # DISPLAY
    # ==================================================================

    # --- Metrics row ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Qmax", f"{qmax:.1f} mL/s", border=True,
              help="Maximum sustained flow rate (\u2265 200 ms)")
    c2.metric("Average Flow", f"{average_flow:.1f} mL/s", border=True,
              help="Total volume \u00f7 voiding time")
    c3.metric("Voiding Time", f"{voiding_time:.1f} s", border=True,
              help="Duration of the detected voiding event")
    c4.metric("Volume", f"{volume_ml:.0f} mL", border=True,
              help="Patient-reported total voided volume")

    if qmax_warning:
        st.warning(qmax_warning)

    # --- Tabs ---
    tab_clinical, tab_details = st.tabs(["Clinical Result", "Processing Details"])

    with tab_clinical:
        st.pyplot(fig4)

    with tab_details:
        with st.expander("Step 1: Signal Conditioning & RMS"):
            st.write(
                f"**Audio:** {diag['duration']:.2f} s &nbsp;|&nbsp; "
                f"**Sample Rate:** {diag['sr']} Hz &nbsp;|&nbsp; "
                f"**Bandpass:** 300 \u2013 2 500 Hz"
            )
            st.pyplot(fig1)

        with st.expander("Step 2: Pre-Processing"):
            st.write(
                f"**Artifacts removed:** {diag['n_artifacts']} frames &nbsp;|&nbsp; "
                f"**Pre-smooth:** Savitzky-Golay (window={savgol_window}, poly=3)"
            )

        with st.expander("Step 3: Event Detection (Hysteresis)"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Noise Window:** {diag['noise_win'][0]:.2f} \u2013 {diag['noise_win'][1]:.2f} s")
                st.write(f"**Noise Floor:** mean={diag['noise_mean']:.5f}, std={diag['noise_std']:.5f}")
                st.write(f"**Onset Threshold (5\u03c3):** {diag['onset_threshold']:.5f}")
                st.write(f"**Offset Threshold (2\u03c3):** {diag['offset_threshold']:.5f}")
            with col_b:
                st.write(f"**Adaptive Hangover:** {diag['hangover']:.1f} s")
                if diag['spectral_ext'] > 0:
                    st.write(f"**Spectral Flux Extension:** +{diag['spectral_ext']:.1f} s")
                if diag['cp_ext'] > 0:
                    st.write(f"**Changepoint Extension:** +{diag['cp_ext']:.1f} s")
            st.pyplot(fig2)

        with st.expander("Step 4: Zero-Floor & Smoothing"):
            st.write(
                f"**Baseline:** offset threshold (2\u03c3) &nbsp;|&nbsp; "
                f"**Smoother:** Gaussian (\u03c3={diag['gauss_sigma']}) &nbsp;|&nbsp; "
                f"**Onset Ramp:** {diag['ramp_duration']:.2f} s"
            )
            st.pyplot(fig3)

    # Close all figures to free memory
    plt.close('all')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if start_button and is_ready:
    process_audio(uploaded_file, total_volume_ml)
elif not is_ready:
    st.info("Upload a WAV file and enter the total voided volume to begin analysis.")

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "**Disclaimer:** This app performs signal processing only. "
    "It does not provide medical diagnoses or constitute clinical advice."
)
