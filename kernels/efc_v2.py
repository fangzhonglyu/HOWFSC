from kernels.kernel import Kernel, rand_tensor
from specs.system_spec.system_spec import SystemSpec
import torch

class EFC(Kernel):

    def __init__(self, data_type, system: SystemSpec):
        super().__init__('EFC', data_type)

        self.M = system.dof
        self.N = system.n_actuators

        self.FLOPs = 2 * self.M * self.N
        self.mem_access = (4 if self.datatype == 'fp32' else 8) * (self.M * self.N)
        self.mem_capacity = (4 if self.datatype == 'fp32' else 8) * (self.M * self.N)

    def run(self, MJT, E):
        return -torch.matmul(MJT, E)

    def setup(self, device):
        MJT = rand_tensor((self.M, self.N), self.datatype, device, name="MJT")
        E = rand_tensor((self.N, 1), self.datatype, device, name="E")

        return MJT, E