raw = """
SRAM Solution	Power (W)	Frequency
	36.10	1
	38.06	10
	42.71	30
	47.81	50
	63.05	100
	108.26	150
	136.32	200
	271.06	300
	791.84	400
	1,661.26	430
	2,640.38	440
	6,490.47	450
		
HBM Solution	Power (W)	Frequency
	28.35960379	1
	59.27261256	10
	132.6347973	30
	213.2089615	50
	454.1551672	100
	798.6210565	150
	1242.46723	200
	3004.18251	300
	4059.14953	330
	5091.75229	350
		
GPU (B200)	Power (W)	Frequency
	100	6.8300
	250	16.7006
	500	32.2231
	1000	60.1999
	2000	104.8778
	4000	165.0811
	8000	228.1474
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
plt.xlim(0,4000)
plt.ylabel("Frequency", fontsize=16)
plt.title("Power vs Frequency Comparison", fontsize=18)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14)
plt.tight_layout()

plt.savefig("plots/generated/power_vs_frequency.pdf")