import numpy as np
import matplotlib.pyplot as plt
from kernels.kernel import Kernel
from specs.compute_spec.compute_specs import ComputeSpec
from specs.system_spec.system_spec import SystemSpec
from utils.units import pretty_SI
from labellines import labelLines

def plot_roofline(kernels: list[Kernel], compute_spec: ComputeSpec):
    ois = []
    names = []
    for kernel in kernels:
        ois.append(kernel.perf(compute_spec)['arithmetic_intensity'])
        names.append(kernel.name)
    FLOPs = (compute_spec.fp32_FLOPs if kernels[0].datatype == 'fp32' else compute_spec.fp64_FLOPs)
    FLOPs /= 1e9  # in GFLOP/s
    bandwidth = compute_spec.mem_bw / 1e9  # in GB/s
    
    oi_vals = np.logspace(-1, 3, 200)  # from 0.01 to 100 FLOP/Byte

    mem_bound = bandwidth * oi_vals      # in GFLOP/s
    comp_bound = np.full_like(oi_vals, FLOPs)
    roof = np.minimum(mem_bound, comp_bound)

    plt.figure(figsize=(8, 6))
    plt.loglog(oi_vals, mem_bound,     '--', linewidth=2,
               label=f'Peak Memory Bandwidth: {pretty_SI(compute_spec.mem_bw)}B/s')
    plt.loglog(oi_vals, comp_bound,    '-',  linewidth=2,
               label=f'Peak Throughput: {pretty_SI(FLOPs*1e9)}FLOP/s')

    labelLines(plt.gca().get_lines(), zorder=2.5, drop_label=True, align=True, fontsize=12, yoffsets=[0000,20000], xoffsets=[-0.8,-45])

        # Ridge point
    ridge_oi = FLOPs / bandwidth
    # plt.axvline(x=ridge_oi, ymin=0, ymax=1/1.5, color='black', linestyle=':', label='OI Ridge Point')

    # Memory bound region
    plt.fill_between([0,ridge_oi], 0, [0,FLOPs], color='lightsteelblue', alpha=0.3)
    plt.text(np.log(ridge_oi)*0.25, FLOPs*0.01, 'Memory Bound', rotation=0, color='steelblue', fontsize=16)

    plt.fill_between([ridge_oi, 1000], 0, [FLOPs,FLOPs], color='moccasin', alpha=0.3)
    plt.text(ridge_oi*2.2, FLOPs*0.01, 'Compute Bound', rotation=0, color='darkorange', fontsize=16)

    # Plot each workload
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    for i, (oi, name) in enumerate(zip(ois, names)):
        perf = min(bandwidth * oi, FLOPs)
        plt.scatter(oi, perf, color=colors[i % len(colors)], s=80,
                    marker=markers[i % len(markers)],
                    label=f'{name}(OI={oi:.2f})')

    # Annotations & styling
    plt.xlim(oi_vals.min(), oi_vals.max())
    plt.ylim(0, FLOPs * 10)
    plt.xlabel('Operational Intensity (FLOP/Byte)', fontsize=14)
    plt.ylabel('Performance (GFLOP/s)', fontsize=14)
    plt.title('Roofline Model', fontsize=18)
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()

    plt.savefig('plots/generated/fig3.svg', dpi=300)


def plot_example_roofline():

    FLOPs = (compute_spec.fp32_FLOPs if kernels[0].datatype == 'fp32' else compute_spec.fp64_FLOPs) * 2.5
    FLOPs /= 1e9  # in GFLOP/s
    bandwidth = compute_spec.mem_bw / 1e9  # in GB/s

    oi_vals = np.linspace(0.01, 200, 1000)  # from 0.01 to 200 FLOP/Byte

    mem_bound = bandwidth * oi_vals    # in GFLOP/s
    comp_bound = np.full_like(oi_vals, FLOPs )

    plt.figure(figsize=(8, 6))
    plt.plot(oi_vals, mem_bound,     '--', linewidth=2,
               label=f'Peak Memory Bandwidth')
    plt.plot(oi_vals, comp_bound,    '-',  linewidth=2,
               label=f'Peak Compute Throughput')
    
    # Ridge point
    ridge_oi = FLOPs / bandwidth
    # plt.axvline(x=ridge_oi, ymin=0, ymax=1/1.5, color='black', linestyle=':', label='OI Ridge Point')

    # Memory bound region
    plt.fill_between([0,ridge_oi], 0, [0,FLOPs], color='green', alpha=0.3)
    plt.text(ridge_oi*0.4, FLOPs*0.2, 'Memory Bound', rotation=0, color='black', fontsize=16)
    # Compute bound region
    plt.fill_between([ridge_oi, 1000], 0, [FLOPs,FLOPs], color='blue', alpha=0.3)
    plt.text(ridge_oi*1.3, FLOPs*0.2, 'Compute Bound', rotation=0, color='black', fontsize=16)
    # plt.fill_between(oi_vals, 0, comp_bound, where=(oi_vals >= ridge_oi), color='orange', alpha=0.3)

    # Annotations & styling
    plt.xlim(oi_vals.min(), oi_vals.max())
    plt.xlabel('Operational Intensity', fontsize=16)
    plt.ylabel('Performance', fontsize=16)
    plt.ylim(0, FLOPs * 1.5)
    plt.xticks([])
    plt.yticks([])
    plt.title('Roofline Model', fontsize=18)
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()

    plt.savefig('plots/generated/fig3.svg', dpi=300)

if __name__ == '__main__':
    from kernels.pwp import PWP
    from kernels.ekf import EKF
    from kernels.efc_v2 import EFC
    from kernels.gain import Gain

    from specs.compute_spec.compute_specs import ComputeSpec

    system_spec = SystemSpec('specs/system_spec/LUVOIR_VIS_A.yml')
    
    compute_spec = ComputeSpec('specs/compute_spec/H100-80G-SXM.yml')
    #compute_spec = ComputeSpec('specs/compute_spec/BAE5545.yml')

    pwp_kernel = PWP('fp32', system_spec)
    kf_probing_kernel = EKF('fp32', system_spec)
    efc_kernel = EFC('fp32', system_spec)
    gain_kernel = Gain('fp32', system_spec)

    kernels = [pwp_kernel, kf_probing_kernel, efc_kernel]

    plot_roofline(kernels, compute_spec)
    # plot_example_roofline()


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