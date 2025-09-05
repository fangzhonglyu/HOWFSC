from specs.compute_spec.compute_specs import ComputeSpec
import torch
import time

def synchronize_device(device):
    """
    Synchronizes the device to ensure all operations are complete.
    """
    if device.type == 'cuda':
        torch.cuda.synchronize(device=device)
    elif device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'npu':
        torch.npu.synchronize(device=device)
    elif device.type == 'musa':
        torch.musa.synchronize(device=device)
    else:
        pass  # No synchronization needed for CPU

def cleanup_device(device):
    """
    Cleans up the device by emptying the cache.
    """
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'npu':
        torch.npu.empty_cache()
    elif device.type == 'musa':
        torch.musa.empty_cache()
    else:
        pass  # No cleanup needed for CPU


def rand_tensor(shape, datatype, device, name=""):
    """
    Allocates a tensor on the specified device with the given shape and datatype.
    """
    if datatype == 'fp32':
        tensor =  torch.randn(shape, dtype=torch.float32, device=device)
    else:
        tensor = torch.randn(shape, dtype=torch.float64, device=device)

    if name:
        print(f"  Allocated tensor '{name}' of shape {shape}, size {tensor.element_size() * tensor.nelement() / 1e9} GB, datatype: {tensor.dtype}, device: {device}")
    else:
        print(f"  Allocated tensor of shape {shape}, size {tensor.element_size() * tensor.nelement() / 1e9} GB, datatype: {tensor.dtype}, device: {device}")
    return tensor

class Kernel():

    def __init__(self, name: str, datatype: str, *args, **kwargs):
        self.name = name
        self.datatype = datatype
        self.FLOPs = 0
        self.mem_access = 0
        self.mem_capacity = 0
    
    def compute_arithmetic_intensity(self):
        if self.mem_access == 0:
            return float('inf')
        return self.FLOPs / self.mem_access

    def get_TFLOPs(self):
        return self.FLOPs / 1e12
    
    def get_mem_access_GB(self):
        return self.mem_access / 1e9
    
    def get_mem_capacity_GB(self):
        return self.mem_capacity / 1e9
    
    def setup(self, device):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError
    
    def sim(self, device, min_time=3.0):
        cleanup_device(device)
        inputs = self.setup(device)
        synchronize_device(device)

        iters = 0
        elapsed_time = 0.0
        start_time = time.perf_counter()

        # Warm-up run
        while elapsed_time < min_time:
            synchronize_device(device)
            self.run(*inputs)
            synchronize_device(device)
            elapsed_time = time.perf_counter() - start_time
            iters += 1

        # Actual timing run
        synchronize_device(device)
        start_time = time.perf_counter()
        for _ in range(iters):
            self.run(*inputs)
        synchronize_device(device)
        elapsed_time = time.perf_counter() - start_time
        
        return elapsed_time / iters
    
    def perf(self, compute: ComputeSpec):
        FLOPs_time = self.FLOPs / (compute.fp32_FLOPs if self.datatype == 'fp32' else compute.fp64_FLOPs)
        mem_time = self.mem_access / compute.mem_bw
        bounding_factor = 'compute' if FLOPs_time > mem_time else 'memory'
        time = max(FLOPs_time, mem_time)
        
        return {
            'time': time,
            'bounding_factor': bounding_factor,
            'FLOPs_time': FLOPs_time,
            'mem_time': mem_time,
            'arithmetic_intensity': self.compute_arithmetic_intensity(),
            'mem_cap_GB': self.mem_capacity / 1e9,
        }
