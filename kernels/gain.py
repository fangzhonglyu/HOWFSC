from kernels.kernel import Kernel, rand_tensor
from specs.system_spec.system_spec import SystemSpec
import torch

class Gain(Kernel):
    """
        Kernel for Precomputing Gain Matrix -> MJT = INV(J.T @ J + α*I) @ J.T

        Args:
            J (torch.Tensor): Jacobian matrix of shape (M, N)
            alpha (float): Regularization parameter
        Params:
            M (int): Total Degree of Freedom = 2 * N_pixel * N_channels
            N (int): Number of DM Actuators 
    """

    def __init__(self, data_type, system: SystemSpec):
        super().__init__('Precompute Gain', data_type)

        self.M = system.dof
        self.N = system.n_actuators

        self.FLOPs = self.M * self.N**2 + (2*self.N**3)/3
        self.mem_access = (4 if self.datatype == 'fp32' else 8) * (self.N * self.M + self.N * self.N)
        self.mem_capacity = (4 if self.datatype == 'fp32' else 8) * (self.M * self.N)

    def run(self, J, alpha):
        JTJ = torch.matmul(J.T, J)
        JTJ.diagonal().add_(alpha)
        return torch.linalg.solve(JTJ, J.T)
    
    def setup(self, device):
        J = rand_tensor((self.M, self.N), self.datatype, device, name="J")
        alpha = 0.1
        return J, alpha