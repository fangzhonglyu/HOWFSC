from kernels.efc import EFC
from specs.compute_spec.compute_specs import ComputeSpec
from specs.system_spec.system_spec import SystemSpec
import time
import torch

def sim(system: SystemSpec, compute:ComputeSpec, Kernel, device='cpu'):
    print(f"Running simulation for {system.name}")
    k = Kernel('fp32', system)
    perf_fp32 = k.perf(compute)
    print(f"Roofline Results for {k.name} with fp32:")
    print(perf_fp32)
    # Start a timer
    run_time = k.sim(device)
    print(f"  Run time: {run_time} seconds")

    return f"{system.name}, {compute.name}, {k.name}, fp32, {perf_fp32['arithmetic_intensity']}, {perf_fp32['time']}, {run_time}, {perf_fp32['bounding_factor']}\n"

res_5090 = sim(SystemSpec('specs/system_spec/WFIRST.yml'), ComputeSpec('specs/compute_spec/RTX5090.yml'), Kernel=EFC, device=torch.device('cuda'))
res_9950x3d = sim(SystemSpec('specs/system_spec/WFIRST.yml'), ComputeSpec('specs/compute_spec/R9-9950X3D.yml'), Kernel=EFC, device=torch.device('cpu'))

with open('results.csv', 'w') as f:
    f.write("system, compute, kernel, datatype, arithmetic_intensity, roofline_time, actual_time, bounding_factor\n")
    f.write(res_5090)
    f.write(res_9950x3d)