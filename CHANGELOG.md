# Sonoflow V2 — Change Log

## Session 3: Robust Onset Ramp (Frame-0 Application)

### Problem
The half-cosine onset ramp was ineffective on recordings where Gaussian smoothing + noise clamping fragmented the onset into short micro-segments. The first non-zero segment could be as few as 8 frames (~0.09s), so `ramp_frames = min(target, seg_len) = 8` — the ramp covered only 0.09s instead of the intended 0.50s, producing a near-vertical jump at t=0.

**Root cause:** `ramp_frames = min(ramp_frames_target, seg_len)` clamped the ramp to the first segment's length.

### Fix: Apply ramp from frame 0 unconditionally
- Removed segment detection loop (`diff_nz`, `seg_starts`, `seg_ends`, `for s, e in zip(...)`)
- Ramp window now applied to `rms_final[:ramp_frames]` starting from frame 0
- `ramp_frames` clamped only by total signal length, not by segment length
- Frames already at 0 stay at 0 (0 × window = 0), preserving leading zeros and micro-gaps
- By the time a real intermittent pause occurs (seconds in), the window value is 1.0 — no attenuation
- Updated comment block and debug diagnostic (removed `seg_start` field)

### What stayed the same
- Ramp duration formula: `np.clip(0.10 * time_to_peak, 0.5, 2.0)`
- Half-cosine window shape: `0.5 * (1 - cos(linspace(0, π, N)))`
- Segment-wise Gaussian smoothing (unchanged — still uses per-segment detection)
- Trailing tail trim, Step 5 normalisation

---

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


## Session 4: Adaptive Cosine Onset Ramp

### Problem
When the noise floor is very clean (std near 0), the backward onset extension from 5-sigma to 2-sigma gains only 1-2 frames. The curve jumps near-vertically from 0 to significant flow at t=0. Real uroflowmetry curves have a gradual onset ramp.

### Fix: Adaptive Half-Cosine Onset Ramp (Step 4)
Applied after segment-wise Gaussian smoothing, before trailing tail trim.

- **Duration** = 10% of time-to-peak, clamped to [0.5 s, 2.0 s]
- **Window** = half-cosine: `0.5 * (1 - cos(linspace(0, pi, N)))` — produces a smooth S-curve from 0 to 1
- **Scope** = only the FIRST non-zero segment (onset) is ramped; internal segments after zero pauses are NOT ramped, preserving intermittency evidence
- `endpoint=True` so window reaches exactly 1.0 at the ramp end (no discontinuity)
- AUC normalization in Step 5 auto-compensates for the small area reduction from ramping

### What Does NOT Change
- Event detection (Step 3) — ramp is post-processing only
- Zero-flooring and noise_std threshold — already applied before ramp
- Segment-wise Gaussian smoothing — untouched
- Trailing tail trim — operates on signal end, no interaction
- Qmax — peak region unaffected by onset ramp


## Current Pipeline Summary (Steps 1-5)

1. **Signal Conditioning & RMS** — bandpass 300-2500 Hz, librosa RMS (frame=1024, hop=512)
2. **Pre-Processing** — Otsu-based artifact removal + Savitzky-Golay smoothing (window=51, poly=3) BEFORE event detection
3. **Event Detection** — hysteresis (5 sigma onset / 2 sigma offset), adaptive hangover 2-8 s, spectral flux validation, PELT changepoint; onset extended backwards to offset_threshold for zero-start
4. **Zero-Floor & Smoothing** — subtract offset_threshold, noise_std threshold, segment-wise Gaussian smoothing (sigma=1.5% of segment), adaptive half-cosine onset ramp (10% of time-to-peak, 0.5-2.0 s), trailing tail trim (2% of peak)
5. **Normalisation & Flow Rate** — AUC volume normalisation, Qmax sustain validation (>= 200 ms at 90% of peak)


## Known Issues / Future Work
- Artifact removal centroid threshold (250 Hz) below bandpass floor (300 Hz) — may rarely trigger
- No handling of multi-void / interrupted flow patterns (distinct voiding episodes)
- Linear RMS-to-flow assumption (non-linear power-law would be more accurate)
- Intermittent flow dips may not fully reach 0 mL/s with current noise_std threshold — under evaluation with additional recordings
- No caching (`@st.cache_data`) for audio loading
- No CSV/PDF export
- requirements.txt has no version pins
