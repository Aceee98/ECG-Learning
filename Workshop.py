# =========================================================
#  ECG SIGNAL PROCESSING WORKSHOP
#  Python equivalent of ECG_Workshop.m
#  Original MATLAB code © 2026 Dr Mahdi Torabi / Torabi Signals Ltd.
#  Python translation for PyCharm use
# =========================================================
#
#  Signal details:
#    - File         : ecg_data.csv
#    - Variable     : channel_1  (65536 samples)
#    - Sampling rate: 256 Hz
#    - Duration     : 256 seconds (~4.3 minutes)
#    - Amplitude    : -0.4 to 1.2 mV (realistic synthetic ECG)
#
#  HOW TO USE:
#    Run each SECTION one at a time (or run the full script).
#    Make sure ecg_data.csv is in the same folder as this script.
# =========================================================

from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import butter, filtfilt

# Use a non-interactive backend if running headless; comment out for PyCharm
# matplotlib.use('TkAgg')   # <- uncomment if plots do not appear

plt.rcParams['figure.facecolor'] = 'white'


# =========================================================
#  SECTION 1: LOAD ECG DATA
# =========================================================

DATA_FILE = Path(__file__).with_name('ecg_data.csv')
csv_data = np.genfromtxt(DATA_FILE, delimiter=',', names=True)
ecg = np.asarray(csv_data['channel_1'], dtype=float).flatten()

fs = 256                                  # Sampling frequency (Hz) — DO NOT CHANGE
N  = len(ecg)                             # Total number of samples
t  = np.arange(N) / fs                   # Time axis in seconds

print('==============================================')
print('  ECG Data Loaded Successfully!')
print('==============================================')
print(f'  File          : {DATA_FILE.name}')
print('  Column        : channel_1')
print(f'  Total samples : {N}')
print(f'  Sampling rate : {fs} Hz')
print(f'  Duration      : {t[-1]:.1f} seconds ({t[-1]/60:.1f} minutes)')
print(f'  Amplitude range: {ecg.min():.2f}  to  {ecg.max():.2f} mV')
print('==============================================')


# =========================================================
#  SECTION 2: PLOT THE FULL ECG SIGNAL
# =========================================================

fig2, ax2 = plt.subplots(figsize=(13, 3.5))
fig2.canvas.manager.set_window_title('SECTION 2 - Full ECG Signal')

ax2.plot(t, ecg, color=(0.1, 0.38, 0.75), linewidth=0.7)
ax2.set_xlabel('Time (seconds)', fontsize=12)
ax2.set_ylabel('Amplitude (mV)', fontsize=12)
ax2.set_title('Full ECG Recording  |  channel_1  |  fs = 256 Hz  |  Duration = 256 s',
              fontsize=13, fontweight='bold')
ax2.set_xlim([0, t[-1]])
ax2.set_ylim([ecg.min() - 0.1, ecg.max() + 0.1])
ax2.grid(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(labelsize=11)
plt.tight_layout()


# =========================================================
#  SECTION 3: ZOOM INTO A 5-SECOND WINDOW
# =========================================================

start_sec = 30         # <-- Change me!  Try 0, 30, 60 ...
end_sec   = start_sec + 5

win5 = (t >= start_sec) & (t < end_sec)

fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(13, 6.5))
fig3.canvas.manager.set_window_title('SECTION 3 - Full ECG + 5-Second Zoom')

# --- Top subplot: full signal with shaded zoom region ---
ax3a.plot(t, ecg, color=(0.65, 0.65, 0.65), linewidth=0.5)
ax3a.add_patch(mpatches.Rectangle(
    (start_sec, ecg.min() - 0.1),
    end_sec - start_sec,
    (ecg.max() + 0.1) - (ecg.min() - 0.1),
    color=(1.0, 0.88, 0.40), alpha=0.40, linewidth=0
))
ax3a.set_xlabel('Time (s)', fontsize=11)
ax3a.set_ylabel('Amplitude (mV)', fontsize=11)
ax3a.set_title('Full ECG  -  Yellow region = zoomed section below',
               fontsize=12, fontweight='bold')
ax3a.set_xlim([0, t[-1]])
ax3a.grid(True)
ax3a.spines['top'].set_visible(False)
ax3a.spines['right'].set_visible(False)
ax3a.tick_params(labelsize=11)

# --- Bottom subplot: 5-second zoomed view ---
ax3b.plot(t[win5], ecg[win5], color=(0.85, 0.18, 0.15), linewidth=1.5)
ax3b.set_xlabel('Time (s)', fontsize=11)
ax3b.set_ylabel('Amplitude (mV)', fontsize=11)
ax3b.set_title(f'Zoomed View:  {start_sec:.0f} - {end_sec:.0f} seconds  (5 seconds window)',
               fontsize=12, fontweight='bold')
ax3b.set_xlim([start_sec, end_sec])
ax3b.grid(True)
ax3b.spines['top'].set_visible(False)
ax3b.spines['right'].set_visible(False)
ax3b.tick_params(labelsize=11)

plt.tight_layout()


# =========================================================
#  SECTION 4: ADD NOISE TO THE ECG
# =========================================================

# ---- CHANGE THESE VALUES to control noise level ----------
snr_wgn       = 10    # <-- SNR for White Gaussian Noise (dB)
powerline_amp = 0.3   # <-- 50 Hz powerline amplitude (mV): try 0.1, 0.3, 0.6
baseline_amp  = 0.3
# ----------------------------------------------------------

# (a) White Gaussian Noise  —  Python equivalent of awgn(..., 'measured')
signal_power_db = 10 * np.log10(np.mean(ecg ** 2))
noise_power_db  = signal_power_db - snr_wgn
noise_std       = np.sqrt(10 ** (noise_power_db / 10))
wgn             = np.random.randn(N) * noise_std
ecg_wgn         = ecg + wgn

# Baseline noise (under 0.2hz)
baseline_wander = baseline_amp * np.sin(2 * np.pi * 0.2 * t)
ecg_baseline    = ecg + baseline_wander

# (b) 50 Hz Powerline Interference
powerline_noise = powerline_amp * np.sin(2 * np.pi * 50 * t)
ecg_powerline   = ecg + powerline_noise





# --- View first 5 seconds ---
view_start = 0;  view_end = 5
vw = (t >= view_start) & (t < view_end)

fig4, axes4 = plt.subplots(4, 1, figsize=(13, 8.5))
fig4.canvas.manager.set_window_title('SECTION 4 - ECG with Two Types of Noise')

axes4[0].plot(t[vw], ecg[vw], color=(0.10, 0.38, 0.75), linewidth=1.4)
axes4[0].set_title('Original Clean ECG', fontsize=12, fontweight='bold')
axes4[0].set_ylabel('Amplitude (mV)', fontsize=10)
axes4[0].set_xlim([view_start, view_end])
axes4[0].grid(True)
axes4[0].spines['top'].set_visible(False)
axes4[0].spines['right'].set_visible(False)

axes4[1].plot(t[vw], ecg_wgn[vw], color=(0.75, 0.10, 0.10), linewidth=0.9)
axes4[1].set_title(f'ECG + White Gaussian Noise   (SNR = {snr_wgn} dB)',
                   fontsize=12, fontweight='bold')
axes4[1].set_ylabel('Amplitude (mV)', fontsize=10)
axes4[1].set_xlim([view_start, view_end])
axes4[1].grid(True)
axes4[1].spines['top'].set_visible(False)
axes4[1].spines['right'].set_visible(False)

axes4[2].plot(t[vw], ecg_powerline[vw], color=(0.45, 0.10, 0.75), linewidth=0.9)
axes4[2].set_title(f'ECG + 50 Hz Powerline Noise   (amplitude = {powerline_amp:.1f} mV)',
                   fontsize=12, fontweight='bold')
axes4[2].set_xlabel('Time (seconds)', fontsize=10)
axes4[2].set_ylabel('Amplitude (mV)', fontsize=10)
axes4[2].set_xlim([view_start, view_end])
axes4[2].grid(True)
axes4[2].spines['top'].set_visible(False)
axes4[2].spines['right'].set_visible(False)

axes4[3].plot(t[vw], ecg_baseline[vw], color=(0.10, 0.55, 0.75), linewidth=0.9)
axes4[3].set_title(f'ECG + Baseline Wander   (amplitude = {baseline_amp:.1f} mV, 0.2 Hz)',
                   fontsize=12, fontweight='bold')
axes4[3].set_xlabel('Time (seconds)', fontsize=10)
axes4[3].set_ylabel('Amplitude (mV)', fontsize=10)
axes4[3].set_xlim([view_start, view_end])
axes4[3].grid(True)
axes4[3].spines['top'].set_visible(False)
axes4[3].spines['right'].set_visible(False)

for ax in axes4:
    ax.tick_params(labelsize=10)
plt.tight_layout()


# =========================================================
#  SECTION 5: PRE-PROCESSING - REMOVE THE NOISE
# =========================================================

# ---- CHANGE THESE to experiment with filter settings -----
bp_low     = 0.5   # <-- Bandpass lower cutoff (Hz)
bp_high    = 40    # <-- Bandpass upper cutoff (Hz)
notch_freq = 50    # <-- Notch filter frequency (Hz) — do not change
notch_bw   = 2     # <-- Notch bandwidth (Hz): try 1, 2, 4
hp_cutoff  = 0.5
# ----------------------------------------------------------

filter_order = 4
nyq          = fs / 2   # Nyquist frequency

# (a) Bandpass filter for WGN  (equivalent to MATLAB butter + filtfilt)
b_bp, a_bp      = butter(filter_order, [bp_low / nyq, bp_high / nyq], btype='bandpass')
ecg_wgn_clean   = filtfilt(b_bp, a_bp, ecg_wgn)

# (b) Notch (bandstop) filter for 50 Hz powerline noise
notch_lo        = (notch_freq - notch_bw / 2) / nyq
notch_hi        = (notch_freq + notch_bw / 2) / nyq
b_notch, a_notch      = butter(filter_order, [notch_lo, notch_hi], btype='bandstop')
ecg_powerline_clean   = filtfilt(b_notch, a_notch, ecg_powerline)

# Baseline wander (High pass)
b_hp, a_hp         = butter(filter_order, hp_cutoff / nyq, btype='high')
ecg_baseline_clean = filtfilt(b_hp, a_hp, ecg_baseline)

# --- Plot ---
view_start = 0;  view_end = 10
vw = (t >= view_start) & (t < view_end)

fig5, axes5 = plt.subplots(3, 2, figsize=(14, 8.5))
fig5.canvas.manager.set_window_title('SECTION 5 - Pre-Processing: Noise Removal')

# Original (spans both columns in row 0)
ax_orig = plt.subplot2grid((3, 2), (0, 0), colspan=2, fig=fig5)
ax_orig.plot(t[vw], ecg[vw], color=(0.10, 0.38, 0.75), linewidth=1.4)
ax_orig.set_title('Original Clean ECG  (reference)', fontsize=11, fontweight='bold')
ax_orig.set_ylabel('Amplitude (mV)', fontsize=10)
ax_orig.set_xlim([view_start, view_end])
ax_orig.grid(True)
ax_orig.spines['top'].set_visible(False)
ax_orig.spines['right'].set_visible(False)

# WGN noisy  (row 1, col 0)
ax_wgn_noisy = plt.subplot2grid((3, 2), (1, 0), fig=fig5)
ax_wgn_noisy.plot(t[vw], ecg_wgn[vw], color=(0.80, 0.20, 0.20), linewidth=0.9)
ax_wgn_noisy.set_title(f'ECG + WGN   (SNR = {snr_wgn} dB)', fontsize=11, fontweight='bold')
ax_wgn_noisy.set_ylabel('Amplitude (mV)', fontsize=10)
ax_wgn_noisy.set_xlim([view_start, view_end])
ax_wgn_noisy.grid(True)
ax_wgn_noisy.spines['top'].set_visible(False)
ax_wgn_noisy.spines['right'].set_visible(False)

# WGN after bandpass  (row 1, col 1)
ax_wgn_clean = plt.subplot2grid((3, 2), (1, 1), fig=fig5)
ax_wgn_clean.plot(t[vw], ecg_wgn_clean[vw], color=(0.10, 0.55, 0.25), linewidth=1.4, label='Filtered')
ax_wgn_clean.plot(t[vw], ecg[vw], '--', color=(0.70, 0.70, 0.70), linewidth=0.9, label='Original')
ax_wgn_clean.set_title(f'WGN After Bandpass Filter  [ {bp_low:.1f} - {bp_high:.0f} Hz ]',
                       fontsize=11, fontweight='bold')
ax_wgn_clean.set_ylabel('Amplitude (mV)', fontsize=10)
ax_wgn_clean.legend(loc='upper right', fontsize=9)
ax_wgn_clean.set_xlim([view_start, view_end])
ax_wgn_clean.grid(True)
ax_wgn_clean.spines['top'].set_visible(False)
ax_wgn_clean.spines['right'].set_visible(False)

# 50 Hz noisy  (row 2, col 0)
ax_pl_noisy = plt.subplot2grid((3, 2), (2, 0), fig=fig5)
ax_pl_noisy.plot(t[vw], ecg_powerline[vw], color=(0.45, 0.10, 0.75), linewidth=0.9)
ax_pl_noisy.set_title(f'ECG + 50 Hz Powerline   ({powerline_amp:.1f} mV)',
                      fontsize=11, fontweight='bold')
ax_pl_noisy.set_xlabel('Time (s)', fontsize=10)
ax_pl_noisy.set_ylabel('Amplitude (mV)', fontsize=10)
ax_pl_noisy.set_xlim([view_start, view_end])
ax_pl_noisy.grid(True)
ax_pl_noisy.spines['top'].set_visible(False)
ax_pl_noisy.spines['right'].set_visible(False)

# 50 Hz after notch  (row 2, col 1)
ax_pl_clean = plt.subplot2grid((3, 2), (2, 1), fig=fig5)
ax_pl_clean.plot(t[vw], ecg_powerline_clean[vw], color=(0.10, 0.55, 0.25), linewidth=1.4, label='Filtered')
ax_pl_clean.plot(t[vw], ecg[vw], '--', color=(0.70, 0.70, 0.70), linewidth=0.9, label='Original')
ax_pl_clean.set_title(f'50 Hz After Notch Filter  [ {notch_freq} Hz +/- {notch_bw/2:.0f} Hz ]',
                      fontsize=11, fontweight='bold')
ax_pl_clean.set_xlabel('Time (s)', fontsize=10)
ax_pl_clean.set_ylabel('Amplitude (mV)', fontsize=10)
ax_pl_clean.legend(loc='upper right', fontsize=9)
ax_pl_clean.set_xlim([view_start, view_end])
ax_pl_clean.grid(True)
ax_pl_clean.spines['top'].set_visible(False)
ax_pl_clean.spines['right'].set_visible(False)

ax_bw_noisy = plt.subplot2grid((4, 2), (3, 0), fig=fig5)
ax_bw_noisy.plot(t[vw], ecg_baseline[vw], color=(0.10, 0.55, 0.75), linewidth=0.9)
ax_bw_noisy.set_title(f'ECG + Baseline Wander   ({baseline_amp:.1f} mV, 0.2 Hz)', fontsize=11, fontweight='bold')
ax_bw_noisy.set_xlabel('Time (s)', fontsize=10)
ax_bw_noisy.set_ylabel('Amplitude (mV)', fontsize=10)
ax_bw_noisy.set_xlim([view_start, view_end])
ax_bw_noisy.grid(True)
ax_bw_noisy.spines['top'].set_visible(False)
ax_bw_noisy.spines['right'].set_visible(False)

ax_bw_clean = plt.subplot2grid((4, 2), (3, 1), fig=fig5)
ax_bw_clean.plot(t[vw], ecg_baseline_clean[vw], color=(0.10, 0.55, 0.25), linewidth=1.4, label='Filtered')
ax_bw_clean.plot(t[vw], ecg[vw], '--', color=(0.70, 0.70, 0.70), linewidth=0.9, label='Original')
ax_bw_clean.set_title(f'Baseline Wander After High-Pass Filter  [ >{hp_cutoff:.1f} Hz ]', fontsize=11, fontweight='bold')
ax_bw_clean.set_xlabel('Time (s)', fontsize=10)
ax_bw_clean.set_ylabel('Amplitude (mV)', fontsize=10)
ax_bw_clean.legend(loc='upper right', fontsize=9)
ax_bw_clean.set_xlim([view_start, view_end])
ax_bw_clean.grid(True)
ax_bw_clean.spines['top'].set_visible(False)
ax_bw_clean.spines['right'].set_visible(False)

for ax in [ax_orig, ax_wgn_noisy, ax_wgn_clean, ax_pl_noisy, ax_pl_clean, ax_bw_noisy, ax_bw_clean]:
    ax.tick_params(labelsize=10)

fig5.suptitle('Pre-Processing: Bandpass Filter (WGN)  |  Notch Filter (50 Hz Powerline)',
              fontsize=13, fontweight='bold')
plt.tight_layout()


# =========================================================
#  SECTION 6: FREQUENCY ANALYSIS - FFT
# =========================================================

# --- Single-sided FFT magnitude  (equivalent to MATLAB (2/N)*abs(fft(x,N))) ---
def get_fft(x):
    return (2 / N) * np.abs(np.fft.fft(x, N))

freq          = np.arange(N) * (fs / N)    # frequency axis in Hz
half          = np.arange(1, N // 2 + 1)   # positive-frequency indices (1-based → 0-based below)
half_idx      = half - 1                   # 0-based numpy indices

fft_orig      = get_fft(ecg)
fft_powerline = get_fft(ecg_powerline)

freq_half     = freq[half_idx]             # positive frequencies

# --- Find heartbeat peak (0.5 – 4 Hz) ---
hr_mask            = (freq_half >= 0.5) & (freq_half <= 4.0)
fft_orig_hr        = fft_orig[half_idx][hr_mask]
hr_loc             = np.argmax(fft_orig_hr)
hr_freq            = freq_half[hr_mask][hr_loc]
hr_bpm             = hr_freq * 60

print(f'Heart Rate from FFT: {hr_freq:.2f} Hz  =  {hr_bpm:.0f} BPM')







# --- 50 Hz spike value ---
idx_50    = np.argmin(np.abs(freq_half - 50))
spike_val = fft_powerline[half_idx][idx_50]

# --- Heartbeat peak value in original FFT ---
hr_idx_global = np.searchsorted(freq_half, hr_freq)
hr_val        = fft_orig[half_idx][hr_idx_global]

# --- Figure ---
fig6 = plt.figure(figsize=(13, 8))
fig6.canvas.manager.set_window_title('SECTION 6 - FFT Frequency Analysis')
fmax = 128

# Top: overlaid comparison
ax6_top = plt.subplot2grid((3, 2), (0, 0), colspan=2, fig=fig6)
ax6_top.plot(freq_half, fft_powerline[half_idx], color=(0.45, 0.10, 0.75), linewidth=1.0,
             label='ECG + 50 Hz Noise')
ax6_top.plot(freq_half, fft_orig[half_idx],      color=(0.10, 0.38, 0.75), linewidth=1.6,
             label='Original ECG (clean)')
ax6_top.axvline(50, color=(0.7, 0.0, 0.7), linestyle='--', linewidth=1.0)
ax6_top.text(50.5, ax6_top.get_ylim()[1] * 0.05, '50 Hz powerline', color=(0.7, 0.0, 0.7), fontsize=9)
ax6_top.set_xlabel('Frequency (Hz)', fontsize=11)
ax6_top.set_ylabel('Magnitude (mV)', fontsize=11)
ax6_top.set_title('FFT Comparison: Original ECG  vs  ECG + 50 Hz Powerline Noise',
                  fontsize=12, fontweight='bold')
ax6_top.legend(loc='upper right', fontsize=10)
ax6_top.set_xlim([0, fmax])
ax6_top.grid(True)
ax6_top.spines['top'].set_visible(False)
ax6_top.spines['right'].set_visible(False)

# Bottom-left: original FFT with heartbeat labelled
ax6_bl = plt.subplot2grid((3, 2), (1, 0), rowspan=2, fig=fig6)
ax6_bl.plot(freq_half, fft_orig[half_idx], color=(0.10, 0.38, 0.75), linewidth=1.6)
ax6_bl.plot(hr_freq, hr_val, 'rv', markersize=10, markerfacecolor='r')
ax6_bl.axvline(hr_freq, color='r', linestyle='--', linewidth=1.2)
ax6_bl.text(hr_freq + 0.3, hr_val * 0.3,
            f'Heartbeat: {hr_freq:.2f} Hz = {hr_bpm:.0f} BPM',
            color='r', fontsize=9)
ax6_bl.set_xlabel('Frequency (Hz)', fontsize=11)
ax6_bl.set_ylabel('Magnitude (mV)', fontsize=11)
ax6_bl.set_title('Original ECG - Heartbeat Peak Identified',
                 fontsize=11, fontweight='bold')
ax6_bl.set_xlim([0, fmax])
ax6_bl.grid(True)
ax6_bl.spines['top'].set_visible(False)
ax6_bl.spines['right'].set_visible(False)



# Bottom-right: powerline FFT with 50 Hz spike labelled
ax6_br = plt.subplot2grid((3, 2), (1, 1), rowspan=2, fig=fig6)
ax6_br.plot(freq_half, fft_powerline[half_idx], color=(0.45, 0.10, 0.75), linewidth=1.0)
ax6_br.plot(50, spike_val, 'v', color=(0.8, 0, 0), markersize=10, markerfacecolor=(0.8, 0, 0))
ax6_br.axvline(50, color='r', linestyle='--', linewidth=1.2)
ax6_br.text(50.5, spike_val * 0.7,
            f'50 Hz Noise Spike  ({spike_val:.3f} mV)',
            color='r', fontsize=9)
ax6_br.set_xlabel('Frequency (Hz)', fontsize=11)
ax6_br.set_ylabel('Magnitude (mV)', fontsize=11)
ax6_br.set_title('ECG + 50 Hz Noise - Spike Clearly Visible at 50 Hz',
                 fontsize=11, fontweight='bold')
ax6_br.set_xlim([0, fmax])
ax6_br.grid(True)
ax6_br.spines['top'].set_visible(False)
ax6_br.spines['right'].set_visible(False)

for ax in [ax6_top, ax6_bl, ax6_br]:
    ax.tick_params(labelsize=10)

fig6.suptitle(f'FFT Analysis  |  Heartbeat = {hr_bpm:.0f} BPM  |  50 Hz Powerline = {powerline_amp:.1f} mV',
              fontsize=13, fontweight='bold')
plt.tight_layout()


# =========================================================
#  SECTION 7: R-PEAK DETECTION  (Pan-Tompkins inspired)
# =========================================================
#
#  What is R-peak detection?
#    The R-peak is the tallest, sharpest spike in each PQRST
#    heartbeat complex. Detecting it accurately lets us:
#      - Count heart rate (BPM)
#      - Compute RR intervals (time between beats)
#      - Spot arrhythmias (irregular rhythms)
#
#  Algorithm steps used here:
#    1. Bandpass filter  (0.5 – 15 Hz)  — isolates QRS energy
#    2. Differentiate   — highlights steep slopes of the R wave
#    3. Square          — amplifies large values, suppresses small
#    4. Moving average  — smooths into broad "bumps" over each beat
#    5. Adaptive threshold — finds peaks above a dynamic level
#    6. Refractory period — ignores detections < 200 ms apart
#       (human heart cannot beat faster than ~300 BPM)
#
#  TODO for participants:
#    1. Run and count the red markers on the zoomed plot.
#       Do they line up with every R peak?
#    2. Look at the RR interval plot — are the intervals regular?
#    3. Multiply mean RR interval by 60 to get mean BPM.
#       Does it match the FFT-based BPM from Section 6?
#
# ----------------------------------------------------------

from scipy.signal import find_peaks

# ------ Step 1: Bandpass filter (0.5 – 15 Hz) for QRS band ------
b_qrs, a_qrs = butter(4, [0.5 / (fs/2), 15.0 / (fs/2)], btype='bandpass')
ecg_bp        = filtfilt(b_qrs, a_qrs, ecg)

# ------ Step 2: Differentiate (emphasises steep R-wave slope) ----
ecg_diff = np.diff(ecg_bp, prepend=ecg_bp[0])

# ------ Step 3: Square (make all values positive, boost large) ---
ecg_sq = ecg_diff ** 2

# ------ Step 4: Moving-average integration (150 ms window) -------
win_ms   = 150                              # integration window in ms
win_samp = int(np.round(win_ms * fs / 1000))  # convert to samples
kernel   = np.ones(win_samp) / win_samp
ecg_mwa  = np.convolve(ecg_sq, kernel, mode='same')

# ------ Step 5 & 6: Adaptive threshold + refractory period -------
threshold      = 0.35 * np.max(ecg_mwa)    # 35 % of global peak
refractory_ms  = 200                        # minimum ms between beats
refractory_samp = int(refractory_ms * fs / 1000)

r_peaks_raw, _ = find_peaks(ecg_mwa,
                             height=threshold,
                             distance=refractory_samp)

# ------ Refine: snap each detection to nearest true maximum ------
#  The MWA peak is slightly delayed vs the actual R spike.
#  Search a ±80 ms window around each candidate in the filtered ECG.
search_rad = int(0.08 * fs)   # 80 ms in samples
r_peaks    = np.array([
    p + np.argmax(ecg_bp[max(0, p - search_rad) : p + search_rad + 1]) - search_rad
    for p in r_peaks_raw
], dtype=int)
r_peaks = np.clip(r_peaks, 0, N - 1)

# ------ Heart-rate statistics from RR intervals ------------------
rr_intervals_samp = np.diff(r_peaks)              # in samples
rr_intervals_sec  = rr_intervals_samp / fs        # in seconds
rr_intervals_ms   = rr_intervals_sec * 1000       # in milliseconds
heart_rates       = 60.0 / rr_intervals_sec       # instantaneous BPM

mean_rr  = np.mean(rr_intervals_ms)
std_rr   = np.std(rr_intervals_ms)
mean_bpm = np.mean(heart_rates)

print()
print('==============================================')
print('  R-Peak Detection Results')
print('==============================================')
print(f'  Total R-peaks detected : {len(r_peaks)}')
print(f'  Mean RR interval       : {mean_rr:.1f} ms  (std = {std_rr:.1f} ms)')
print(f'  Mean Heart Rate        : {mean_bpm:.1f} BPM')
print(f'  Min Heart Rate         : {heart_rates.min():.1f} BPM')
print(f'  Max Heart Rate         : {heart_rates.max():.1f} BPM')
print('==============================================')

# =============================================================
#  FIGURE 7a: Pan-Tompkins Processing Pipeline (first 10 s)
# =============================================================
view_s = 0;  view_e = 10
vp = (t >= view_s) & (t < view_e)

fig7a, axes7a = plt.subplots(4, 1, figsize=(13, 9))
fig7a.canvas.manager.set_window_title('SECTION 7a - R-Peak Detection: Processing Pipeline')

# Original ECG
axes7a[0].plot(t[vp], ecg[vp], color=(0.10, 0.38, 0.75), linewidth=1.2)
axes7a[0].set_ylabel('Amplitude (mV)', fontsize=9)
axes7a[0].set_title('Step 1 — Original ECG (clean)', fontsize=10, fontweight='bold')
axes7a[0].set_xlim([view_s, view_e]);  axes7a[0].grid(True)
axes7a[0].spines['top'].set_visible(False);  axes7a[0].spines['right'].set_visible(False)

# Bandpass filtered
axes7a[1].plot(t[vp], ecg_bp[vp], color=(0.10, 0.60, 0.30), linewidth=1.2)
axes7a[1].set_ylabel('Amplitude (mV)', fontsize=9)
axes7a[1].set_title('Step 2 — Bandpass Filtered  (0.5 – 15 Hz)', fontsize=10, fontweight='bold')
axes7a[1].set_xlim([view_s, view_e]);  axes7a[1].grid(True)
axes7a[1].spines['top'].set_visible(False);  axes7a[1].spines['right'].set_visible(False)

# Squared signal
axes7a[2].plot(t[vp], ecg_sq[vp], color=(0.85, 0.45, 0.05), linewidth=1.0)
axes7a[2].set_ylabel('Amplitude²', fontsize=9)
axes7a[2].set_title('Step 3 — Differentiated & Squared', fontsize=10, fontweight='bold')
axes7a[2].set_xlim([view_s, view_e]);  axes7a[2].grid(True)
axes7a[2].spines['top'].set_visible(False);  axes7a[2].spines['right'].set_visible(False)

# Moving-average + threshold + detected peaks
r_in_view = r_peaks[(t[r_peaks] >= view_s) & (t[r_peaks] < view_e)]
axes7a[3].plot(t[vp], ecg_mwa[vp], color=(0.55, 0.10, 0.70), linewidth=1.2, label='MWA signal')
axes7a[3].axhline(threshold, color='orange', linestyle='--', linewidth=1.2, label=f'Threshold ({threshold:.4f})')
axes7a[3].plot(t[r_in_view], ecg_mwa[r_in_view], 'rv', markersize=9,
               markerfacecolor='red', label='R-peak detected', zorder=5)
axes7a[3].set_xlabel('Time (s)', fontsize=10)
axes7a[3].set_ylabel('Amplitude', fontsize=9)
axes7a[3].set_title('Step 4 — Moving-Window Integration  +  Threshold  +  Detected Peaks',
                    fontsize=10, fontweight='bold')
axes7a[3].legend(loc='upper right', fontsize=9)
axes7a[3].set_xlim([view_s, view_e]);  axes7a[3].grid(True)
axes7a[3].spines['top'].set_visible(False);  axes7a[3].spines['right'].set_visible(False)

for ax in axes7a:
    ax.tick_params(labelsize=9)

fig7a.suptitle('Section 7 — R-Peak Detection: Pan-Tompkins Processing Pipeline',
               fontsize=13, fontweight='bold')
plt.tight_layout()


# =============================================================
#  FIGURE 7b: R-peaks overlaid on ECG  (full + 10 s zoom)
# =============================================================
fig7b, (ax7b_top, ax7b_bot) = plt.subplots(2, 1, figsize=(13, 7))
fig7b.canvas.manager.set_window_title('SECTION 7b - R-Peak Locations on ECG')

# --- Top: full signal with all R-peaks ---
ax7b_top.plot(t, ecg, color=(0.10, 0.38, 0.75), linewidth=0.6, label='ECG', zorder=2)
ax7b_top.plot(t[r_peaks], ecg[r_peaks], 'r^',
              markersize=5, markerfacecolor='red', markeredgewidth=0.3,
              label=f'R-peaks  (n = {len(r_peaks)})', zorder=3)
ax7b_top.set_xlabel('Time (s)', fontsize=11)
ax7b_top.set_ylabel('Amplitude (mV)', fontsize=11)
ax7b_top.set_title(f'Full ECG with All R-Peaks Marked  '
                   f'|  {len(r_peaks)} beats  |  Mean HR = {mean_bpm:.0f} BPM',
                   fontsize=12, fontweight='bold')
ax7b_top.set_xlim([0, t[-1]])
ax7b_top.legend(loc='upper right', fontsize=10)
ax7b_top.grid(True)
ax7b_top.spines['top'].set_visible(False);  ax7b_top.spines['right'].set_visible(False)

# --- Bottom: 10-second zoom with larger markers and labels ---
z_s = 10;  z_e = 20
zp  = (t >= z_s) & (t < z_e)
r_zoom = r_peaks[(t[r_peaks] >= z_s) & (t[r_peaks] < z_e)]

ax7b_bot.plot(t[zp], ecg[zp], color=(0.10, 0.38, 0.75), linewidth=1.5, zorder=2)
ax7b_bot.plot(t[r_zoom], ecg[r_zoom], 'r^',
              markersize=11, markerfacecolor='red', markeredgewidth=0.5,
              label='R-peak', zorder=3)

# Label each peak with its index number
for i, rp in enumerate(r_zoom):
    beat_num = np.searchsorted(r_peaks, rp) + 1
    ax7b_bot.annotate(f'#{beat_num}',
                      xy=(t[rp], ecg[rp]),
                      xytext=(0, 12), textcoords='offset points',
                      ha='center', fontsize=8, color='darkred', fontweight='bold')

ax7b_bot.set_xlabel('Time (s)', fontsize=11)
ax7b_bot.set_ylabel('Amplitude (mV)', fontsize=11)
ax7b_bot.set_title(f'Zoomed View:  {z_s} – {z_e} s  |  Each R-peak labelled with beat number',
                   fontsize=12, fontweight='bold')
ax7b_bot.set_xlim([z_s, z_e])
ax7b_bot.legend(loc='upper right', fontsize=10)
ax7b_bot.grid(True)
ax7b_bot.spines['top'].set_visible(False);  ax7b_bot.spines['right'].set_visible(False)

for ax in [ax7b_top, ax7b_bot]:
    ax.tick_params(labelsize=10)

fig7b.suptitle('Section 7 — R-Peak Detection Results', fontsize=13, fontweight='bold')
plt.tight_layout()


# =============================================================
#  FIGURE 7c: RR Intervals & Instantaneous Heart Rate
# =============================================================
beat_times = t[r_peaks[1:]]     # time of each beat (from beat 2 onward)

fig7c, (ax7c_top, ax7c_bot) = plt.subplots(2, 1, figsize=(13, 6))
fig7c.canvas.manager.set_window_title('SECTION 7c - RR Intervals & Heart Rate')

# RR interval over time
ax7c_top.plot(beat_times, rr_intervals_ms, color=(0.10, 0.38, 0.75),
              linewidth=1.4, marker='o', markersize=3, label='RR interval')
ax7c_top.axhline(mean_rr, color='red', linestyle='--', linewidth=1.2,
                 label=f'Mean = {mean_rr:.1f} ms')
ax7c_top.fill_between(beat_times,
                      mean_rr - std_rr, mean_rr + std_rr,
                      color='red', alpha=0.12, label=f'±1 SD ({std_rr:.1f} ms)')
ax7c_top.set_xlabel('Time (s)', fontsize=11)
ax7c_top.set_ylabel('RR Interval (ms)', fontsize=11)
ax7c_top.set_title(f'RR Intervals Over Time  |  Mean = {mean_rr:.1f} ms  |  SD = {std_rr:.1f} ms',
                   fontsize=12, fontweight='bold')
ax7c_top.legend(loc='upper right', fontsize=10)
ax7c_top.set_xlim([0, t[-1]])
ax7c_top.grid(True)
ax7c_top.spines['top'].set_visible(False);  ax7c_top.spines['right'].set_visible(False)

# Instantaneous heart rate
ax7c_bot.plot(beat_times, heart_rates, color=(0.75, 0.10, 0.10),
              linewidth=1.4, marker='o', markersize=3, label='Instantaneous HR')
ax7c_bot.axhline(mean_bpm, color='darkred', linestyle='--', linewidth=1.2,
                 label=f'Mean = {mean_bpm:.1f} BPM')
ax7c_bot.set_xlabel('Time (s)', fontsize=11)
ax7c_bot.set_ylabel('Heart Rate (BPM)', fontsize=11)
ax7c_bot.set_title(f'Instantaneous Heart Rate  |  Mean = {mean_bpm:.1f} BPM  '
                   f'|  Range: {heart_rates.min():.0f} – {heart_rates.max():.0f} BPM',
                   fontsize=12, fontweight='bold')
ax7c_bot.legend(loc='upper right', fontsize=10)
ax7c_bot.set_xlim([0, t[-1]])
ax7c_bot.grid(True)
ax7c_bot.spines['top'].set_visible(False);  ax7c_bot.spines['right'].set_visible(False)

for ax in [ax7c_top, ax7c_bot]:
    ax.tick_params(labelsize=10)

fig7c.suptitle('Section 7 — RR Intervals & Instantaneous Heart Rate Tachogram',
               fontsize=13, fontweight='bold')
plt.tight_layout()


# =========================================================
#  WORKSHOP COMPLETE!
# =========================================================

print()
print('==============================================')
print('  ECG Workshop - All Sections Complete!')
print('==============================================')
print(f'  Section 1: Loaded channel_1  ({N} samples, {fs} Hz)')
print(f'  Section 2: Full ECG plotted  ({t[-1]:.0f} seconds)')
print('  Section 3: 5-second zoom window')
print(f'  Section 4: WGN (SNR={snr_wgn} dB)  +  50 Hz Powerline ({powerline_amp:.1f} mV)')
print(f'  Section 5: Bandpass [{bp_low:.1f}-{bp_high:.0f} Hz] + Notch [{notch_freq} Hz] filters applied')
print('  Section 6: FFT analysis complete')
print(f'  Section 7: R-peak detection — {len(r_peaks)} beats, {mean_bpm:.0f} BPM mean HR')
print('==============================================')

plt.show()
