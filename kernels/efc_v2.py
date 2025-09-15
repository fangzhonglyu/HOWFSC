from kernels.kernel import Kernel, rand_tensor, one_tensor
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

    def set_MJT(self, MJT):
        self.MJT = MJT

    def run(self, E):
        return -torch.matmul(self.MJT, E)

    def setup(self, device):
        self.MJT = one_tensor((self.N, self.M), self.datatype, device, name="MJT")
        E = rand_tensor((self.M, 1), self.datatype, device, name="E")
        return (E,)