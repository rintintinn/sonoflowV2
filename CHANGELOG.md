# Sonoflow V2 — Change Log

## Session 1: Initial Code Review & End-of-Flow Detection

### Code Review Findings
- **Duplicate code block** (lines 216-221) — removed during rewrite
- **Artifact removal cascading bug** — `rms_clean[i]` was modified in-place while reading neighbours from the same array, causing cascading suppression. Fixed by reading neighbours from the original `rms` array.
- **Centroid threshold** (250 Hz) below bandpass floor (300 Hz) — noted, not yet addressed
- **Linear RMS-to-flow assumption** — noted as limitation; non-linear power-law would be more accurate

### End-of-Flow Detection Overhaul (7 Recommendations)
**Problem:** Premature cutoff of dribble/tail flow. The original single-threshold approach missed the dribbling phase of urination.

1. **Hysteresis thresholding (Schmitt trigger)**
   - Separate onset (5 sigma) and offset (2 sigma) thresholds
   - Reason: higher onset threshold gives confident detection; lower offset threshold captures the full decay including dribble

2. **Redesigned hangover timer**
   - Timer now resets when signal exceeds the lower offset threshold (not onset threshold)
   - Reason: prevents premature cutoff when dribble briefly dips below onset threshold

3. **Smoothing before event detection**
   - Savitzky-Golay filter (window=51, poly=3) applied BEFORE onset/offset detection
   - Reason: stabilises the noisy dribble signal so threshold comparisons are reliable
   - Trim still uses `rms_clean` (original energy); detection uses `rms_smoothed_full`

4. **Lower zero-floor baseline**
   - Zero-floor subtraction uses offset_threshold (2 sigma) instead of the old 4 sigma
   - Reason: preserves more of the low-amplitude tail signal

5. **Adaptive hangover duration (2-8 s)**
   - Hangover scaled by signal decay rate: 1/e time x 2, clamped to [2, 8] seconds
   - Reason: short voids get short hangover; long slow decays get longer hangover

6. **Spectral flux validation**
   - Uses `librosa.onset.onset_strength` to search up to 10 s beyond initial offset
   - Checks 0.5 s windows for elevated RMS + flux spikes (>= 2 above 3 sigma)
   - Reason: detects impulsive dribble sounds that the RMS threshold might miss

7. **PELT changepoint detection (ruptures library)**
   - L2 cost model as second opinion on offset boundary
   - Extends offset if the segment between old and new offset has mean > 1.5 sigma
   - Reason: statistical changepoint detection catches gradual transitions that threshold methods miss


## Session 2: Plot Smoothing & UI Redesign

### Gaussian Smoothing (Step 4)
**Problem:** Savitzky-Golay filter produced jagged curves with visible high-frequency noise. The clinical goal is to show the trend of urine flow, not raw RMS fluctuations.

- Replaced Savitzky-Golay final smoothing with `scipy.ndimage.gaussian_filter1d`
- Sigma = 1.5% of signal length, `mode='constant'`, `cval=0.0`
- Reason: Gaussian kernel exactly preserves AUC (normalised convolution), has no edge artifacts with constant padding, and produces inherently smooth output in both time and frequency domains

### UI Redesign
- Computation wrapped in `st.status()` for live progress feedback
- Metrics row: `st.columns(4)` + `st.metric(border=True)` for Qmax, Avg Flow, Void Time, Volume
- Two tabs: "Clinical Result" (polished flow chart) + "Processing Details" (expanders for Steps 1-4)
- Custom CSS for metric card styling (blue left border, larger font)
- Clinical matplotlib rcParams: white background, no top/right spines, subtle grid, colour palette [blue, red, teal, gold, grey]
- Created `.streamlit/config.toml` with light clinical theme (primary #0066CC)


## Session 3: Onset, Intermittency & Voiding Time Corrections

### 1. Onset Extension — Flow Starts at Zero
**Problem:** At time 0, the flow curve started above zero (~0.0001 amplitude) because onset was detected at the 5 sigma crossing point, where RMS energy is already well above the noise floor.

**Fix:** After detecting onset at 5 sigma (for confidence), walk backwards to find where the signal was at or below offset_threshold (2 sigma). Since zero-floor subtraction removes that same 2 sigma, the extended start point becomes exactly 0 mL/s. Gaussian smoothing then creates a natural ramp-up.

- Location: after PELT changepoint detection, before trimming
- The 5 sigma detection is preserved for robustness; we only extend the trim window earlier

### 2. Segment-Wise Gaussian Smoothing — Preserve Intermittency
**Problem:** Applying Gaussian smoothing to the entire signal bridged over zero-flow pauses, erasing evidence of intermittent flow (a clinically significant pattern).

**Fix:** Smooth each contiguous non-zero segment independently. Zeros between segments are untouched.

- Find contiguous non-zero runs using diff of boolean mask
- Each segment gets its own Gaussian kernel (sigma = 1.5% of segment length)
- `mode='constant'`, `cval=0.0` — edges taper naturally to zero
- Zero regions (flow pauses) remain at exactly 0

### 3. Noise Floor Threshold — Clean Pause Boundaries
**Problem:** After zero-floor subtraction (subtracting 2 sigma), tiny residual values in the pause regions prevented clean zero gaps. Segment-wise smoothing treated the entire signal as one segment.

**Fix:** Force values below `noise_std` (1 sigma) to exactly zero after zero-flooring.

- Rationale: after subtracting offset_threshold (noise_mean + 2 sigma), a residual below noise_std means the original signal was between 2 sigma and 3 sigma above noise mean — still noise variability, not real flow
- Creates clean zero gaps so segment-wise smoothing correctly splits flow bursts

### 4. Trailing Tail Trim — Accurate Voiding Time
**Problem:** Offset detection (hysteresis + spectral flux + PELT) extended the voiding window to ~44 s, but actual flow ended around 29-30 s. The signal decays gradually through the threshold without a clean zero gap, so post-flow ambient noise micro-blips inflated the voiding time.

**First attempt (reverted):** Gap-based trim — walk backwards looking for trailing segments preceded by >= 2 s zero gaps. Did not work because the signal decays gradually with no clean zero gap.

**Current implementation:** Amplitude-based trim — define "effective end of flow" as the last point where amplitude >= 2% of peak, then continue to the next zero crossing for a clean cutoff. Micro-blips beyond this point are excluded.

- Location: after segment-wise smoothing, before Step 4 diagnostic plot
- Re-trims both `rms_final` and `times_trimmed`

### Changes Considered but Reverted
- **Gap interpolation across pauses:** Linearly interpolated across internal zero gaps before smoothing. Reverted because it erased intermittent flow evidence, which is clinically significant.
- **Smoothed-signal pause mask:** Used `rms_smoothed_full` to identify dip regions and zero them in the raw floored signal. Reverted at user's preference — the noise_std threshold version was preferred.


## Current Pipeline Summary (Steps 1-5)

1. **Signal Conditioning & RMS** — bandpass 300-2500 Hz, librosa RMS (frame=1024, hop=512)
2. **Pre-Processing** — Otsu-based artifact removal + Savitzky-Golay smoothing (window=51, poly=3) BEFORE event detection
3. **Event Detection** — hysteresis (5 sigma onset / 2 sigma offset), adaptive hangover 2-8 s, spectral flux validation, PELT changepoint; onset extended backwards to offset_threshold for zero-start
4. **Zero-Floor & Smoothing** — subtract offset_threshold, noise_std threshold, segment-wise Gaussian smoothing (sigma=1.5% of segment), trailing tail trim (2% of peak)
5. **Normalisation & Flow Rate** — AUC volume normalisation, Qmax sustain validation (>= 200 ms at 90% of peak)


## Known Issues / Future Work
- Artifact removal centroid threshold (250 Hz) below bandpass floor (300 Hz) — may rarely trigger
- No handling of multi-void / interrupted flow patterns (distinct voiding episodes)
- Linear RMS-to-flow assumption (non-linear power-law would be more accurate)
- Intermittent flow dips may not fully reach 0 mL/s with current noise_std threshold — under evaluation with additional recordings
- No caching (`@st.cache_data`) for audio loading
- No CSV/PDF export
- requirements.txt has no version pins
