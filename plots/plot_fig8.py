raw = """
SRAM Solution	Power (W)	Frequency
	36.23	1
	91.35543999	10
	47.19	30
	56.03	50
	84.65	100
	153.54	150
	229.38	200
	1,016.74	300
	1,400.76	310
	5,349.71	330
	17,338.03	335
		
		
		
		
HBM Solution	Power (W)	Frequency
	31.48687333	1
	91.1792658	10
	233.7333056	30
	393.8646536	50
	897.3946031	100
	1685.379495	150
	3042.436083	200
	6305.216814	250
	10682.64196	270
		
		
		
		
		
		
GPU (B200)	Power (W)	Frequency
	1000	35.5204
	2000	70.3673
	4000	136.9799
	8000	228.2312
	16000	277.0678
"""

import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Parsing logic
# ----------------------------
datasets = {}
current = None

def clean_num(s):
    return float(s.replace(",", ""))

for line in raw.splitlines():
    line = line.strip()
    if not line:
        continue

    # Section header
    if "Solution" in line or "GPU" in line:
        current = line.split("\t")[0]
        datasets[current] = {"power": [], "freq": []}
        continue

    # Data rows: expect two numeric values
    parts = line.split("\t")
    if len(parts) >= 2:
        try:
            power = clean_num(parts[0])
            freq = clean_num(parts[1])
            datasets[current]["power"].append(power)
            datasets[current]["freq"].append(freq)
        except:
            pass

# ----------------------------
# Plot
# ----------------------------
sns.set_palette("Set2")
plt.figure(figsize=(8,6))

markers = ["o", "s", "D"]  # circle, square, diamond
for (label, data), marker in zip(datasets.items(), markers):
    plt.plot(
        data["power"], data["freq"],
        marker=marker,
        linewidth=2,
        markersize=7,
        label=label
    )

plt.xlabel("Power (W)", fontsize=16)
plt.xlim(0,10000)
plt.ylabel("Frequency", fontsize=16)
plt.title("Power vs Frequency Comparison", fontsize=18)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14)
plt.tight_layout()

plt.savefig("plots/generated/power_vs_frequency.pdf")