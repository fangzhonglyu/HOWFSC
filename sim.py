from kernels.efc_v2 import EFC
from kernels.gain import Gain
from kernels.pwp import PWP
from specs.compute_spec.compute_specs import ComputeSpec
from specs.system_spec.system_spec import SystemSpec
import time
import torch
import platform

def sim(system: SystemSpec, compute:ComputeSpec, Kernel, device='cpu', datatype='fp32'):
    print(f"Running simulation for {system.name}")
    k = Kernel(datatype, system)
    perf = k.perf(compute)
    print(f"Roofline Results for {k.name} with {datatype}:")
    print(perf)
    # Start a timer
    run_time = k.sim(device)
    print(f"  Run time: {run_time} seconds")

    return f"{system.name}, {compute.name}, {k.name}, {datatype}, {perf['arithmetic_intensity']}, {perf['time']}, {run_time}, {perf['bounding_factor']}\n"


# res_dual_9575f = sim(SystemSpec('specs/system_spec/LUVOIR_VIS_A.yml'), ComputeSpec('specs/compute_spec/Dual-EPYC-9575F.yml'), Kernel=PWP, device=torch.device('cpu'), datatype='fp32')
rtx_5090 = sim(SystemSpec('specs/system_spec/WFIRST.yml'), ComputeSpec('specs/compute_spec/RTX5090.yml'), Kernel=EFC, device=torch.device('cuda'), datatype='fp32')

# Run only if ubuntu:
# if platform.system() == 'Linux':
#     res_5090 = sim(SystemSpec('specs/system_spec/WFIRST.yml'), ComputeSpec('specs/compute_spec/RTX5090.yml'), Kernel=EFC, device=torch.device('cuda'), datatype='fp32')
#     res_9950x3d = sim(SystemSpec('specs/system_spec/WFIRST.yml'), ComputeSpec('specs/compute_spec/R9-9950X3D.yml'), Kernel=EFC, device=torch.device('cpu'), datatype='fp32')

#     with open('results_ubuntu.csv', 'w') as f:
#         f.write("system, compute, kernel, datatype, arithmetic_intensity, roofline_time, actual_time, bounding_factor\n")
#         f.write(res_5090)
#         f.write(res_9950x3d)

# # Run only if macOS:
# if platform.system() == 'Darwin':
#     res_m3max = sim(SystemSpec('specs/system_spec/WFIRST.yml'), ComputeSpec('specs/compute_spec/M3-MAX.yml'), Kernel=Gain, device=torch.device('mps'), datatype='fp32')

#     with open('results_macos.csv', 'w') as f:
#         f.write("system, compute, kernel, datatype, arithmetic_intensity, roofline_time, actual_time, bounding_factor\n")
#         f.write(res_m3max)
