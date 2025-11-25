import numpy as np
import matplotlib.pyplot as plt
from kernels.kernel import Kernel
from specs.compute_spec.compute_specs import ComputeSpec
from specs.system_spec.system_spec import SystemSpec
from utils.units import pretty_SI
from labellines import labelLines

from seaborn import color_palette
# set2


def plot_perf(compute_specs: list[ComputeSpec]):
    colors = color_palette("Set2", n_colors=10)
    plt.figure(figsize=(8, 6))
    for compute_spec in compute_specs:
        plt.scatter(compute_spec.fp64_FLOPs/1e9, compute_spec.mem_bw/1e9, 
                    label=compute_spec.name, s=50, marker='s')
    
    for i, txt in enumerate([cs.name for cs in compute_specs]):
        plt.annotate(txt, (compute_specs[i].fp64_FLOPs/1e9, compute_specs[i].mem_bw/1e9),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=13)

    plt.xlabel('FP64 FLOPs (GFLOPs)',fontsize=14)
    plt.ylabel('Memory Bandwidth (GB/s)',fontsize=14)
    plt.yscale('log')
    plt.ylim(1, 3e5)
    plt.xscale('log')
    plt.xlim(1, 1e6)
    plt.title('Modern Hardware Landscape',fontsize=18)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/generated/fig4.svg', dpi=300)

import os

if __name__ == '__main__':
    compute_specs = []
    # glob OS directory for yml files
    spec_dir = 'specs/compute_spec/'
    for filename in os.listdir(spec_dir):
        if filename.endswith('.yml'):
            compute_spec = ComputeSpec(os.path.join(spec_dir, filename))
            compute_specs.append(compute_spec)
    plot_perf(compute_specs)


# def plot_roofline(ais, names, bandwidth, peak_flops):
#     """
#     Draw a roofline model with multiple workloads.

#     Parameters
#     ----------
#     ais : list of float
#         Arithmetic intensities of each workload (FLOP/Byte).
#     names : list of str
#         Names corresponding to each workload.
#     bandwidth : float
#         Memory bandwidth in GB/s.
#     peak_flops : float
#         Peak compute throughput in GFLOP/s.
#     """
#     # Generate a range of AI values for plotting (log scale)
#     ai_vals = np.logspace(-1, 2, 200)  # from 0.01 to 100 FLOP/Byte

#     # Compute the two roofline bounds
#     mem_bound = bandwidth * ai_vals      # in GFLOP/s
#     comp_bound = np.full_like(ai_vals, peak_flops)
#     roof = np.minimum(mem_bound, comp_bound)

#     # Set up the plot
#     plt.figure(figsize=(8, 6))
#     plt.loglog(ai_vals, mem_bound,     '--', linewidth=2,
#                label=f'Memory bound: {bandwidth} GB/s')
#     plt.loglog(ai_vals, comp_bound,    '-',  linewidth=2,
#                label=f'Compute bound: {peak_flops} GFLOP/s')
#     # plt.loglog(ai_vals, roof,          '-.', linewidth=2, color='gray',
#     #            label='Roof')

#     # Plot each workload
#     colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#     for i, (ai, name) in enumerate(zip(ais, names)):
#         perf = min(bandwidth * ai, peak_flops)
#         plt.scatter(ai, perf, color=colors[i % len(colors)], s=80,
#                     label=f'{name}: AI={ai:.2f}, Perf={perf:.1f} GF/s')

#     # Annotations & styling
#     plt.xlim(ai_vals.min(), ai_vals.max())
#     plt.xlabel('Arithmetic Intensity (FLOP / Byte)', fontsize=12)
#     plt.ylabel('Performance (GFLOP / s)',      fontsize=12)
#     plt.title('Roofline Model',                fontsize=14)

#     plt.legend(loc='lower right', fontsize=9)

#     plt.savefig('plots/generated/fig3.svg', dpi=300)

# if __name__ == '__main__':
#     # === Replace these with your workloads ===
#     ais = [3.50, 0.25, 0.98]                # AI values for each workload
#     names = ['Pairwise Probing', 'KF with Probing', 'EFC']  # Corresponding workload names
#     bandwidth = 6.75  # Memory bandwidth in GB/s
#     peak_flops = 192 # Peak compute throughput in GFLOP/s

#     plot_roofline(ais, names, bandwidth, peak_flops)