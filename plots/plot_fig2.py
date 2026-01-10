import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as path_effects
from specs.system_spec.system_spec import SystemSpec
from utils.units import pretty_bytes, pretty_bits

LUVOIR_SPEC = SystemSpec('specs/system_spec/LUVOIR_VIS_A.yml')

# ——— Parameters ———
image_size = LUVOIR_SPEC.n_pixels * LUVOIR_SPEC.n_channels * LUVOIR_SPEC.bit_depth / 8
print(f"Image size: {image_size / 1e6} MB")
control_size = LUVOIR_SPEC.n_actuators * 8 # 8 bytes per actuator (fp64)
print(f"Control vector size: {control_size / 1e3} KB")

data_size = image_size + control_size
print(f"Total data size: {data_size / 1e6} MB")

log_distances  = np.linspace(1, 6.5, 200) # distances from 1 km to 1 Million km

distances = 10 ** log_distances # in km
latencies = distances / 300000.0 + 0.001 # speed of light = 300,000 km/s

bandwidths = np.array([1e6, 1e7, 1e8, 1e9, 1e10, 2e11, 1e18]) # in Bytes/second
bandwidths_labels = [pretty_bits(bw) + '/s' for bw in bandwidths]
bandwidths_labels[-1] = 'inf Gb/s'

# ——— Compute iteration rate ———
plt.figure(figsize=(8, 6))
for i, bw in enumerate(bandwidths):
    rtt = 2 * latencies + data_size * 8 / bw # round-trip time: 2 * latency + transfer time
    iters = 1.0 / rtt                    # iterations per second (Hz)
    plt.plot(
        distances,
        iters,
        label=bandwidths_labels[i]
    )
plt.xscale('log')
plt.xlabel("Distance (km)",fontsize=16)
plt.ylabel("Control Frequency (Hz)",fontsize=16)
plt.yscale('log')
plt.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
plt.text(10000, 110, '100 Hz', fontsize=12, va='bottom', ha='right', color='gray')
plt.title("b) Control Freq Limit vs Transfer Distance", fontsize=18, pad=20)
# plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(title="Data Rate", fontsize=12, title_fontsize=12, loc='upper right')
plt.tight_layout()
plt.savefig('plots/generated/fig2.svg', dpi=300)

# ISO_LEVELS = [0.1, 1, 10, 100]
# ISO_LABELS = { i : f'{i} Hz' for i in ISO_LEVELS }

# # ——— Plot heatmap ———
# plt.figure(figsize=(8, 6))
# # convert axes to more-readable units
# X = D # km
# Y = B / 1e6  # MB/s
# cp = plt.contourf(X, Y, iters_capped, levels=50, cmap='viridis', extend='max')
# cs = plt.contour(
#     X, Y, iters_capped,
#     levels=ISO_LEVELS,
#     colors='white',
#     linestyles='--',
#     linewidths=1,
# )

# labels = plt.clabel(
#     cs,
#     fmt=ISO_LABELS,
#     inline=False,
#     fontsize=8,
#     colors='white'
# )

# # Add dark outline to improve contrast and visibility
# for txt in labels:
#     txt.set_path_effects([
#         path_effects.Stroke(linewidth=3, foreground='black'),
#         path_effects.Normal()
#     ])

# cbar = plt.colorbar(cp, label='Iterations per second (Hz)')
# cbar.ax.yaxis.label.set_size(16)
# plt.xlabel('Distance (km)',fontsize=16)
# plt.ylabel('Bandwidth (MB/s)',fontsize=16)
# plt.yscale('log')
# plt.xscale('log')
# plt.title('Control Freq. vs Link Bandwidth & Latency', fontsize=18, pad=20)
# plt.tight_layout()
# plt.savefig('plots/generated/transfer.svg', dpi=300)


# # =======================================

# import numpy as np
# import matplotlib.pyplot as plt

# # ——— Parameters ———
# flop_per_iter = 0.0529*1e12           # work per iteration in FLOPs (e.g. 1e12 for 1 TFLOP)
# compute_tflops = np.linspace(0.5, 60, 200)   # TFLOPs of compute performance (0.5 … 100)
# latencies = np.linspace(0.1e-3, 10e-3, 200)  # one-way latency in seconds (0.1 ms … 10 ms)

# # ——— Compute iteration rate ———
# C, L = np.meshgrid(compute_tflops, latencies)
# compute_perf = C * 1e12        # convert TFLOPs → FLOP/s
# iter_time = 2 * L + flop_per_iter / compute_perf
# iters_per_sec = 1.0 / iter_time

# threshold = 100.0
# iters_capped = np.minimum(iters_per_sec, threshold)

# # ——— Plot heatmap ———
# plt.figure(figsize=(8, 6))
# X = C           # TFLOPs
# Y = L * 1e3     # ms
# cp= plt.contourf(
#     C, L*1e3, iters_capped,
#     levels=50,
#     cmap='viridis',
#     extend='max'         # everything above levels.max() gets the "overflow" color
# )
# cs = plt.contour(
#     C, L * 1e3, iters_per_sec,
#     levels=[100],
#     colors='black',
#     linestyles='--',
#     linewidths=2
# )
# plt.plot(6.87, 1, 'ko')  # Example point, black
# plt.text(6.87, 1, '  7TB/s', color='black', fontsize=10, va='bottom', ha='left')
# plt.clabel(cs, fmt={100: '100 Hz'}, inline=True, fontsize=10, colors='black')
# plt.colorbar(cp, label='Iterations per second (Hz)')
# plt.xlabel('Compute performance (TFLOPs)')
# plt.ylabel('One-way latency (ms)')
# plt.title('Max Control Frequency vs Compute & Latency')
# plt.tight_layout()
# plt.show()