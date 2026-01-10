from kernels.kernel import Kernel
from specs.system_spec.system_spec import SystemSpec
import torch

class EFC(Kernel):

    def __init__(self, data_type, system: SystemSpec):
        super().__init__('EFC', data_type)

        self.M = system.dof
        self.N = system.n_actuators

        self.FLOPs = 2 * self.M * self.N + 2 * self.N * self.N
        self.mem_access = (4 if self.datatype == 'fp32' else 8) * (self.M * (self.M + self.N))
        self.mem_capacity = (4 if self.datatype == 'fp32' else 8) * (self.M * self.N + self.N * self.N)

    def run(self, JT):
        return 
    
    def setup(self, device):
        JT = torch.randn((self.N, self.M), dtype=torch.float32 if self.datatype == 'fp32' else torch.float64, device=device)
        return JT