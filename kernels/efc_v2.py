from kernels.kernel import Kernel, rand_tensor, one_tensor
from specs.system_spec.system_spec import SystemSpec
import torch

class EFC(Kernel):
    """
        Electric Field Conjugation Kernel -> Δp = -MJT @ E
        
        Args:
            MJT (torch.Tensor): Precomputed matrix INV(J.T @ J + α*I) @ J.T of shape (N, M)
            E (torch.Tensor): Tensor of shape (M)
        Params:
            M (int): Total Degree of Freedom = 2 * N_pixel * N_channels
            N (int): Number of DM Actuators
    """

    def __init__(self, data_type, system: SystemSpec):
        super().__init__('EFC', data_type)

        self.M = system.dof
        self.N = system.n_actuators

        self.FLOPs = 2 * self.M * self.N
        self.mem_access = (4 if self.datatype == 'fp32' else 8) * (self.M * self.N)
        self.mem_capacity = (4 if self.datatype == 'fp32' else 8) * (self.M * self.N + self.N * self.N)

    def run(self, M, JT, E, out):
        return torch.matmul(M,torch.matmul(JT, E))

    def setup(self, device):
        M = rand_tensor((self.N, self.N), self.datatype, device, name="M")
        JT = rand_tensor((self.N, self.M), self.datatype, device, name="JT")
        E = rand_tensor((self.M, 1), self.datatype, device, name="E")
        out = rand_tensor((self.N, 1), self.datatype, device, name="out")
        return M, JT, E, out
