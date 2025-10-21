import numpy as np
import matplotlib.pyplot as plt

# Given parameters
A = 1           # Amplitude
f1 = 260        # Frequency 1 (Hz)
f2 = 250        # Frequency 2 (Hz)

# Calculate beat frequency and average frequency
beat_frequency = abs(f1 - f2)
average_frequency = (f1 + f2) / 2

print(f"Given Parameters:")
print(f"Amplitude (A) = {A}")
print(f"Frequency 1 (f1) = {f1} Hz")
print(f"Frequency 2 (f2) = {f2} Hz")
print(f"\nCalculated Values:")
print(f"Beat Frequency = |f1 - f2| = |{f1} - {f2}| = {beat_frequency} Hz")
print(f"Average Frequency = (f1 + f2)/2 = ({f1} + {f2})/2 = {average_frequency} Hz")

# Time array - let's show a few beat cycles
# Beat period = 1/beat_frequency
beat_period = 1 / beat_frequency
t = np.linspace(0, 2 * beat_period, 2000)  # Show 2 complete beat cycles

# Calculate the individual waves
# y1 = A * sin(2π * f1 * t)
# y2 = A * sin(2π * f2 * t)
y1 = A * np.sin(2 * np.pi * f1 * t)
y2 = A * np.sin(2 * np.pi * f2 * t)

# Resultant wave (superposition)
y_resultant = y1 + y2

# Theoretical envelope (beat pattern)
# The envelope amplitude varies as 2A * cos(2π * (f1-f2)/2 * t)
envelope_upper = 2 * A * np.abs(np.cos(2 * np.pi * (f1 - f2) / 2 * t))
envelope_lower = -envelope_upper

# Create the visualization - Only Resultant Wave
plt.figure(figsize=(12, 6))

# Plot only the resultant wave
plt.plot(t, y_resultant, 'b-', linewidth=2, label='Resultant Wave (y1 + y2)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title(f'Beat Phenomenon: Resultant Wave\nf1 = {f1} Hz, f2 = {f2} Hz, Beat Frequency = {beat_frequency} Hz')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print(f"\nBeat Analysis:")
print(f"Beat Period = 1/f_beat = 1/{beat_frequency} = {1/beat_frequency:.3f} seconds")
print(f"In {2 * beat_period:.3f} seconds, we observe {2} complete beat cycles")
maxima_times = []
for i in range(1, len(y_resultant)-1):
    if y_resultant[i] > y_resultant[i-1] and y_resultant[i] > y_resultant[i+1] and y_resultant[i] > 1.5:
        maxima_times.append(t[i])

if len(maxima_times) > 1:
    observed_beat_period = np.mean(np.diff(maxima_times))
    observed_beat_freq = 1 / observed_beat_period
    print(f"Observed beat frequency from maxima: {observed_beat_freq:.1f} Hz")

print(f"\nWave Equations:")
print(f"y1 = {A} × sin(2π × {f1} × t)")
print(f"y2 = {A} × sin(2π × {f2} × t)")
print(f"y_resultant = y1 + y2")
print(f"\nBeat envelope amplitude = 2A × |cos(2π × (f1-f2)/2 × t)|")
print(f"                        = 2 × {A} × |cos(2π × {(f1-f2)/2} × t)|")
