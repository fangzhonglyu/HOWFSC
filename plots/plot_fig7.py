SRAM = """
SRAM Solution	Power (W)	Area (mm2)
62.99		
SRAM	28.49	22184.45902
Interconnect	6.31E-02	72.8
EFC Compute	9.50	9.82E+00
GP Compute	25.00	1600
"""

HBM = """
HBM Solution	Power (W)	Area (mm2)
454.09		
HBM	3.91E+02	1573
HBM PHY	2.84E+01	114.816
Interconnect	6.31E-02	72.8
EFC Compute	9.50	9.82E+00
GP Compute	25.00	1600
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
})

# -------------------------------------------------------------
# Parsing function that handles multi-word names
# -------------------------------------------------------------
def parse_solution(text):
    lines = [l.strip() for l in text.strip().split("\n")]
    entries = []

    for line in lines[2:]:     # skip header + total
        if not line.strip():
            continue
        parts = line.split()
        power = float(parts[-2])
        area  = float(parts[-1])
        name  = " ".join(parts[:-2])
        entries.append((name, power, area))

    return entries


sram_data = parse_solution(SRAM)
hbm_data  = parse_solution(HBM)


# -------------------------------------------------------------
# Function to produce vertically stacked subplots
# -------------------------------------------------------------
def make_plot(data, title, filename):
    sns.set_palette("Set2")
    colors = sns.color_palette("Set2", len(data))

    labels = [x[0] for x in data]
    power_vals = np.array([x[1] for x in data])
    area_vals  = np.array([x[2] for x in data])

    # Convert to percentage
    pcts_power = power_vals / power_vals.sum()
    pcts_area  = area_vals / area_vals.sum()

    fig, axs = plt.subplots(2, 1, figsize=(7, 2.5))

    # ---------------- POWER subplot ----------------
    left = 0
    for lbl, val, col in zip(labels, pcts_power, colors):
        axs[0].barh(
            y=[0], width=[val], left=left, color=col,
            height=0.05               # ← SLIM BAR
        )
        left += val

    axs[0].set_title("Power Breakdown",fontsize=16)
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(-0.02, 0.02)
    axs[0].set_yticks([])            # remove y-axis
    axs[0].set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    axs[0].set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    axs[0].tick_params(axis='x', labelsize=12)

    for spine in axs[0].spines.values():
        spine.set_visible(False)

    # Legend ABOVE plot
    axs[0].legend(
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 2),
        ncol=3,
        fontsize=14,
        frameon=False
    )

    # ---------------- AREA subplot ----------------
    left = 0
    for lbl, val, col in zip(labels, pcts_area, colors):
        axs[1].barh(
            y=[0], width=[val], left=left, color=col,
            height=0.02             # SLIM
        )
        left += val

    axs[1].set_title("Area Breakdown",fontsize=16)
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(-0.01, 0.01)
    axs[1].set_yticks([])
    axs[1].set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    axs[1].set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    axs[1].tick_params(axis='x', labelsize=12)

    for spine in axs[1].spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(hspace=10, top=0.70)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()



# -------------------------------------------------------------
# Generate images
# -------------------------------------------------------------
make_plot(sram_data, "SRAM Solution", "plots/generated/srambd.pdf")
make_plot(hbm_data,  "HBM Solution",  "plots/generated/hbmbd.pdf")