"""


THIS DOCUMENT WAS MOSTLY AI GENERATED, I DID NOT WRITE THIS CODE.
I EDITED THE CODE, I EXPERIMENTED WITH THE CODE, BUT I DID NOT WRITE THE INITAL CODE FOR THIS CODEBASE.


"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path


csv_path = Path(__file__).resolve().parent / 'example.csv'
if not csv_path.exists() or csv_path.stat().st_size == 0:
    raise FileNotFoundError(
        f'CSV file not found or is empty: {csv_path!s}.\n'
        'Make sure example.csv exists in the same folder as example.py.'
    )

try:
    csv_data = np.genfromtxt(csv_path, delimiter=',', names=True)
    if csv_data.size == 0:
        raise ValueError('No data loaded from CSV.')
    column_name = 'channel_1' if 'channel_1' in csv_data.dtype.names else csv_data.dtype.names[0]
    ecg = np.asarray(csv_data[column_name], dtype=float).flatten()
except (IndexError, ValueError, TypeError, OSError):
    try:
        raw = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        if raw.ndim == 1:
            ecg = raw.astype(float)
        else:
            ecg = raw[:, 0].astype(float)
    except Exception as load_err:
        raise RuntimeError(
            f'Failed to load ECG data from {csv_path!s}.\n'
            'Check that example.csv exists, is not empty, and has numeric values.'
        ) from load_err

# Signal information
fs = 256                    # sampling rate (256 measurements per second)
N = len(ecg)                # total number of samples
duration = N / fs           # how long is the recording?
t = np.arange(N) / fs       # time axis in seconds

print("=" * 60)
print("ECG DATA LOADED")
print("=" * 60)
print(f"Samples:           {N}")
print(f"Duration:          {duration:.2f} seconds")
print(f"Sampling rate:     {fs} Hz")
print(f"Min amplitude:     {ecg.min():.3f} mV")
print(f"Max amplitude:     {ecg.max():.3f} mV")
print()

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(t, ecg, color='#1F60C4', linewidth=1.5)

ax.set_xlabel('Time (seconds)', fontsize=11)
ax.set_ylabel('Amplitude (mV)', fontsize=11)
ax.set_title('Full ECG Recording', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

start_sec = 0
end_sec = 3

# Find samples in this window
window_mask = (t >= start_sec) & (t < end_sec)
t_zoom = t[window_mask]
ecg_zoom = ecg[window_mask]

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(t_zoom, ecg_zoom, color='#1F60C4', linewidth=2)

ax.set_xlabel('Time (seconds)', fontsize=11)
ax.set_ylabel('Amplitude (mV)', fontsize=11)
ax.set_title(f'Zoomed View ({start_sec}–{end_sec} seconds)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()


peaks, properties = find_peaks(ecg, height=0.9, distance=fs//2)

print("=" * 60)
print("PEAK DETECTION")
print("=" * 60)
print(f"Number of peaks found: {len(peaks)}")
print(f"Peak locations (samples): {peaks}")
print(f"Peak times (seconds): {t[peaks]}")
print()


fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(t, ecg, color='#1F60C4', linewidth=1.5, label='ECG Signal')
ax.plot(t[peaks], ecg[peaks], 'ro', markersize=8, label='Detected Peaks (R-peaks)')

ax.set_xlabel('Time (seconds)', fontsize=11)
ax.set_ylabel('Amplitude (mV)', fontsize=11)
ax.set_title('ECG with Detected R-Peaks', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()


if len(peaks) > 1:
    # Calculate RR intervals (time between beats)
    rr_intervals = np.diff(t[peaks])  # in seconds
    rr_intervals_ms = rr_intervals * 1000  # convert to milliseconds
    
    # Calculate heart rate
    # Heart rate = 60 / RR interval (in seconds)
    heart_rates = 60 / rr_intervals
    
    avg_hr = np.mean(heart_rates)
    min_hr = np.min(heart_rates)
    max_hr = np.max(heart_rates)
    
    print("=" * 60)
    print("HEART RATE ANALYSIS")
    print("=" * 60)
    print(f"Number of heartbeats: {len(peaks)}")
    print(f"Average RR interval:  {np.mean(rr_intervals_ms):.1f} ms")
    print(f"Average heart rate:   {avg_hr:.1f} BPM")
    print(f"Min heart rate:       {min_hr:.1f} BPM")
    print(f"Max heart rate:       {max_hr:.1f} BPM")
    print()
else:
    print("Not enough peaks to calculate heart rate.")

