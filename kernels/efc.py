from kernels.kernel import Kernel
from specs.system_spec.system_spec import SystemSpec
import torch
import time


from kernels.kernel import cleanup_device, synchronize_device

ITER = 100


class EFC(Kernel):

    def __init__(self, data_type, system: SystemSpec):
        super().__init__('EFC', data_type)

        self.M = system.dof
        self.N = system.n_actuators

        self.FLOPs = 2 * self.M * self.N + 2 * self.M * self.N
        self.mem_access = (4 if self.datatype == 'fp32' else 8) * (self.M * self.M + self.M * self.N)
        self.mem_capacity = (4 if self.datatype == 'fp32' else 8) * (self.M * self.N)

    def run(self, M, JT, E):
        JTE = torch.matmul(JT, E)
        NMJTE = -torch.matmul(M, JTE)
        return NMJTE

    def sim(self, device):
        cleanup_device(device)
        JT = torch.randn((self.M, self.N), dtype=torch.float32 if self.datatype == 'fp32' else torch.float64, device=device)
        M = torch.randn((self.M, self.M), dtype=torch.float32 if self.datatype == 'fp32' else torch.float64, device=device)
        E = torch.randn((self.N, 1), dtype=torch.float32 if self.datatype == 'fp32' else torch.float64, device=device)
        start_time = time.perf_counter()
        for _ in range(ITER):
            synchronize_device(device)
            self.run(M, JT, E)
            synchronize_device(device)
        elapsed_time = time.perf_counter() - start_time
        return elapsed_time / ITER