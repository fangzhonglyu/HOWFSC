from specs.compute_spec.compute_specs import ComputeSpec
import torch

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

    def run(self, *args, **kwargs):
        raise NotImplementedError
    
    def sim(self):
        raise NotImplementedError
    
    def perf(self, compute: ComputeSpec):
        FLOPs_time = self.FLOPs / compute.FLOPs
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
