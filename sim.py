from kernels.efc import EFC
from specs.compute_spec.compute_specs import ComputeSpec
from specs.system_spec.system_spec import SystemSpec
import time
import torch

def sim(system: SystemSpec, Kernel, device='cpu'):
    print(f"Running simulation for {system.name}")
    k = Kernel('fp32', system)
    perf_fp32 = k.perf(ComputeSpec('specs/compute_spec/M3_Max.yml'))
    print(f"Roofline Results for {k.name} with fp32:")
    print(perf_fp32)

    # Start a timer
    elapsed_time = k.sim(device)
    print(f"  Elapsed time: {elapsed_time} seconds")


sim(SystemSpec('specs/system_spec/WFIRST.yml'), Kernel=EFC, device=torch.device('mps'))