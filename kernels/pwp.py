from kernels.kernel import Kernel, rand_tensor
from specs.system_spec.system_spec import SystemSpec
import torch

class PWP(Kernel):
    """
        Per-pixel Pairwise Probing Kernel -> E

        Args:
            I_pairs (torch.Tensor): Probe image pairs of shape (N, 2, N_pixels)
            delta_p (torch.Tensor): Probe commands of shape (N, N_pixels)
    """

    def __init__(self, data_type, system: SystemSpec):
        super().__init__('PWP', data_type)

        self.M = system.dof
        self.N = system.n_actuators

        self.FLOPs = 2 * self.M * self.N
        self.mem_access = (4 if self.datatype == 'fp32' else 8) * (self.M * self.N)
        self.mem_capacity = (4 if self.datatype == 'fp32' else 8) * (self.M * self.N)

    def run(self, I_pairs, delta_p):
        """
        Per-pixel Pairwise Probing Kernel
        
        Args:
            I_pairs (torch.Tensor): Tensor of shape (N, 2, N_pixels)
            delta_p (torch.Tensor): Tensor of shape (N, H, W)
        """
        return -torch.matmu

    def setup(self, device):
        MJT = rand_tensor((self.M, self.N), self.datatype, device, name="MJT")
        E = rand_tensor((self.N, 1), self.datatype, device, name="E")

        return MJT, E