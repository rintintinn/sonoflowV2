import streamlit as st
import numpy as np
import scipy.signal as signal
import librosa
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import scipy.integrate

# Set page config
st.set_page_config(page_title="Acoustic Uroflowmetry Analyzer", layout="wide")

st.title("Acoustic Uroflowmetry Analysis")
st.markdown("Upload a WAV file of urine flow to generate clinically relevant charts.")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Audio (WAV)", type=["wav"])
total_volume_ml = st.sidebar.number_input("Total Voided Volume (mL)", min_value=1, value=None, step=1, format="%d")

# Start Button Logic
# Check if inputs are valid
is_ready = (uploaded_file is not None) and (total_volume_ml is not None)

start_button = st.sidebar.button("Start Processing", disabled=not is_ready)

# --- Processing Pipeline ---

def process_audio(file, volume_ml):
    # Step 1: Signal Conditioning & RMS
    st.header("Step 1: Signal Conditioning & RMS")
    
    # Load Audio
    # librosa.load resamples to 22050 by default, let's keep original sr or at least 44100/48000 if checking up to 2.5kHz
    # target sr=None loads native, often 44.1k or 48k. 
    y, sr = librosa.load(file, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    
    st.write(f"**Audio Loaded:** {duration:.2f} seconds, Sample Rate: {sr} Hz")

    # Bandpass Filter (300Hz - 2500Hz)
    # 4th order Butterworth
    sos = signal.butter(4, [300, 2500], btype='bandpass', fs=sr, output='sos')
    y_filtered = signal.sosfiltfilt(sos, y)

    # RMS Calculation
    frame_length = 1024
    hop_length = 512
    rms = librosa.feature.rms(y=y_filtered, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Graph 1: Raw RMS
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(times, rms, label='Raw RMS')
    ax1.set_title("Raw RMS Amplitude vs. Time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)
    st.pyplot(fig1)

    # Step 2: Event Detection (Smart Noise Floor & Symmetric Threshold)
    st.header("Step 2: Event Detection (Smart Noise Floor & Symmetric Threshold)")

    # 1. Locate "Flow Event" (Otsu Trigger)
    # Use Otsu's Method to find the main loud event (Trigger)
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
    
    # Trigger lies where signal definitely becomes flow
    trigger_indices = np.where(rms > otsu_thresh)[0]
    if len(trigger_indices) == 0:
        st.error("No significant flow detected (Otsu trigger failed).")
        return
    trigger_idx = trigger_indices[0]
    
    # 2. Find Stable Noise Floor
    # Search the region BEFORE the loud event
    # Slide 200ms window across this pre-event silence
    frames_200ms = int(0.2 * sr / hop_length) 
    if frames_200ms < 1: frames_200ms = 1
    
    if trigger_idx < frames_200ms:
        st.warning("Trigger very early. Using initial frames as noise baseline.")
        best_win_start = 0
        best_win_end = min(len(rms), frames_200ms)
    else:
        # Search for Minimum Standard Deviation
        search_limit = trigger_idx
        min_std = float('inf')
        best_win_start = 0
        
        for start_i in range(0, search_limit - frames_200ms + 1):
             end_i = start_i + frames_200ms
             window_seg = rms[start_i:end_i]
             curr_std = np.std(window_seg)
             if curr_std < min_std:
                 min_std = curr_std
                 best_win_start = start_i
        
        best_win_end = best_win_start + frames_200ms

    # 3. Calibrate Unified Threshold (Waterline)
    noise_window = rms[best_win_start:best_win_end]
    noise_mean = np.mean(noise_window)
    noise_std = np.std(noise_window)
    event_threshold = noise_mean + 4 * noise_std
    
    st.write(f"**Stable Noise Window:** {times[best_win_start]:.2f}s - {times[best_win_end]:.2f}s")
    st.write(f"**Calibration:** Mean={noise_mean:.5f}, Std={noise_std:.5f}, Waterline Threshold={event_threshold:.5f}")

    # 4. Detect Onset
    # Scan forward from the end of the stable noise window
    onset_idx = -1
    for i in range(best_win_end, len(rms)):
        if rms[i] > event_threshold:
            onset_idx = i
            break
            
    if onset_idx == -1:
        st.error("Could not detect onset based on calibrated threshold.")
        return
        
    # 5. Detect Offset (Wait-and-See)
    # From the peak flow, scan forward.
    # When RMS < Event_Threshold, start 4s timer.
    # Re-trigger: If RMS > Event_Threshold within 4s, reset.
    
    if onset_idx < len(rms):
        peak_relative_idx = np.argmax(rms[onset_idx:])
        peak_idx = onset_idx + peak_relative_idx
    else:
        peak_idx = onset_idx 
        
    offset_idx = len(rms) - 1
    frames_buffer = int(4.0 * sr / hop_length) # 4 seconds
    
    silence_start_idx = -1
    timer = 0
    
    for i in range(peak_idx, len(rms)):
        if rms[i] < event_threshold:
            if silence_start_idx == -1:
                silence_start_idx = i
            timer += 1
            if timer >= frames_buffer:
                # Timer Expired: Cut at start of silence
                offset_idx = silence_start_idx
                break
        else:
            # Signal > Threshold: Reset
            timer = 0
            silence_start_idx = -1
            
    final_onset = onset_idx
    final_offset = offset_idx
    
    # Trim Signal
    rms_trimmed = rms[final_onset : final_offset+1]
    times_trimmed = times[final_onset : final_offset+1] - times[final_onset]
    
    # Graph 2
    # Display context (noise window -> offset)
    plot_start_idx = max(0, best_win_start - int(1.0 * sr/hop_length)) 
    plot_end_idx = min(len(rms), final_offset + int(1.0 * sr/hop_length))
    
    plot_rms = rms[plot_start_idx : plot_end_idx]
    plot_times = times[plot_start_idx : plot_end_idx] - times[final_onset] # Onset = 0
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(plot_times, plot_rms, label='Signal')
    
    # Shade Stable Noise
    noise_start_t = times[best_win_start] - times[final_onset]
    noise_end_t = times[best_win_end] - times[final_onset]
    ax2.axvspan(noise_start_t, noise_end_t, color='gray', alpha=0.3, label='Stable Noise Window')
    
    # Lines
    ax2.axvline(0.0, color='g', linestyle='--', label='True Onset (0s)')
    offset_t = times[final_offset] - times[final_onset]
    ax2.axvline(offset_t, color='r', linestyle='--', label='Offset')
    
    ax2.axhline(event_threshold, color='orange', linestyle=':', label='Event Threshold (Waterline)')
    
    ax2.set_title("Smart Event Detection: Symmetric Threshold")
    ax2.set_xlabel("Time (s) [Onset = 0]")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)
    
    # Recalculate trimmed for next steps
    rms_trimmed = rms[final_onset : final_offset+1]
    times_trimmed = times[final_onset : final_offset+1] - times[final_onset]
    
    # Recalculate trimmed for next steps
    rms_trimmed = rms[final_onset : final_offset+1]
    times_trimmed = times[final_onset : final_offset+1] - times[final_onset]

    # Step 3: Zero-Floor Subtraction & Smoothing
    st.header("Step 3: Zero-Floor Subtraction & Smoothing")
    
    # Load separate spectral features on the whole file then trim, or compute on trimmed?
    # Need alignment. Best to compute on full then trim.
    # Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y_filtered, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    
    # Artifact Logic
    rms_clean = rms.copy()
    
    # Iterating full RMS first
    for i in range(len(rms_clean)):
        if rms_clean[i] > otsu_thresh and cent[i] < 250: # High RMS but Low Centroid (<250Hz)
            # Replace with neighbor min or interpolate
            # Simple approach: look at left/right valid
            # For simplicity: Use min of +/- 5 neighbors
            start_k = max(0, i-5)
            end_k = min(len(rms_clean), i+6)
            # exclude self if possible, or just take min of window
            local_min = np.min(rms_clean[start_k:end_k])
            rms_clean[i] = local_min

    # Now trim clean RMS
    rms_clean_trimmed = rms_clean[final_onset : final_offset+1]

    # Subtraction & Clipping (Zero-Floor)
    rms_floored = rms_clean_trimmed - event_threshold
    rms_floored = np.maximum(rms_floored, 0)
    
    # Smoothing (Savitzky-Golay)
    # window_length=51, polyorder=3
    # Ensure window_length <= len(rms)
    savgol_window = 51
    if len(rms_floored) < savgol_window:
        savgol_window = len(rms_floored) 
        if savgol_window % 2 == 0: savgol_window -= 1 # Must be odd
        
    if savgol_window > 3:
        rms_smoothed = signal.savgol_filter(rms_floored, window_length=savgol_window, polyorder=3)
    else:
        rms_smoothed = rms_floored

    # Make strictly positive (smoothing might introduce slight negatives near zero)
    rms_smoothed = np.maximum(rms_smoothed, 0)

    # Graph 3
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(times_trimmed, rms_smoothed, 'g-', label='Smoothed Flow')
    ax3.set_title("Zero-Floored & Smoothed Flow (Noise Removed)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude")
    ax3.set_ylim(bottom=0) # Must start at 0
    ax3.grid(True)
    st.pyplot(fig3)

    # Step 4: Normalization & Flow Rate
    st.header("Step 4: Normalization & Flow Rate")
    
    # Integration (AUC)
    # trapz integration over time
    auc = scipy.integrate.trapezoid(rms_smoothed, times_trimmed)
    
    if auc == 0:
        st.error("AUC is zero. Cannot normalize.")
        return

    K = volume_ml / auc
    
    flow_rate = rms_smoothed * K # mL/s
    
    # Qmax Validation
    # "Identify peak flow (Qmax). Verify it is sustained... within 90% of peak height for at least 200ms."
    # 200ms in frames
    frames_200ms = int(0.2 * sr / hop_length)
    if frames_200ms < 1: frames_200ms = 1

    # Find peaks
    # We can just iterate sorted peaks
    peaks, _ = signal.find_peaks(flow_rate)
    # Filter/Sort peaks by height
    if len(peaks) > 0:
        sorted_peaks = sorted(peaks, key=lambda p: flow_rate[p], reverse=True)
        
        valid_qmax_idx = -1
        valid_qmax_val = 0
        
        for p in sorted_peaks:
            val = flow_rate[p]
            thresh_90 = 0.9 * val
            
            # Check sustain defined as: duration where signal >= thresh_90 around peak
            # We can leverage contiguous regions above threshold
            # Find range around p
            above_ids = np.where(flow_rate >= thresh_90)[0]
            
            # Find contiguous block containing p
            # split consecutive
            splits = np.split(above_ids, np.where(np.diff(above_ids) != 1)[0]+1)
            
            sustain_duration = 0
            for group in splits:
                if p in group:
                    # frames count
                    duration_frames = len(group)
                    # or better: time diff
                    # duration_s = (times_trimmed[group[-1]] - times_trimmed[group[0]])
                    # Actually prompt says "remain... for at least 200ms"
                    # frames is safer if time steps are uniform
                    if duration_frames >= frames_200ms:
                        valid_qmax_idx = p
                        valid_qmax_val = val
                    break
            
            if valid_qmax_idx != -1:
                break
        
        if valid_qmax_idx == -1:
            # Fallback to max peak if no sustained peak found (or as prompt says "search next", if none found, usually take absolute max and warn?)
            # Prompt implies "If raw peak is transient... search for NEXT highest".
            # If we run out of peaks, take the global max?
            valid_qmax_idx = np.argmax(flow_rate)
            valid_qmax_val = flow_rate[valid_qmax_idx]
            st.warning("No peak met the 200ms sustain criteria. Using absolute max.")
            
    else:
        valid_qmax_idx = np.argmax(flow_rate)
        valid_qmax_val = flow_rate[valid_qmax_idx]

    # Metrics
    qmax = valid_qmax_val
    average_flow = volume_ml / (times_trimmed[-1] - times_trimmed[0])
    voiding_time = times_trimmed[-1] - times_trimmed[0] # Approx flow time

    # Graph 4: Final Output
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.plot(times_trimmed, flow_rate, 'b-', linewidth=2, label='Flow Rate')
    
    # Qmax line
    ax4.axhline(qmax, color='r', linestyle='--', label=f'Qmax: {qmax:.1f} mL/s')
    
    ax4.set_title("Uroflowmetry: Flow Rate vs. Time")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Flow Rate (mL/s)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Annotation
    stats_text = (
        f"Qmax: {qmax:.1f} mL/s\n"
        f"Avg Flow: {average_flow:.1f} mL/s\n"
        f"Void Time: {voiding_time:.1f} s\n"
        f"Total Vol: {volume_ml:.0f} mL"
    )
    # Place text in box
    ax4.text(0.02, 0.95, stats_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    st.pyplot(fig4)

    # Output Data
    st.success("Processing Complete.")

if start_button and is_ready:
    process_audio(uploaded_file, total_volume_ml)
elif not is_ready:
    if uploaded_file is None:
        st.info("Please upload a WAV file.")
    if total_volume_ml is None:
        st.info("Please enter the Total Voided Volume.")
