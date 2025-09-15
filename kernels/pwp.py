from kernels.kernel import Kernel, rand_tensor
from specs.system_spec.system_spec import SystemSpec
import torch

class PWP(Kernel):
    """
        Per-pixel Pairwise Probing Kernel -> E

        Args:
            I_plus  (torch.Tensor): Positive Probe Images (N_pixels * N_channels, N_probe)
            I_minus (torch.Tensor): Negative Probe Images (N_pixels * N_channels, N_probe)
            delta_p (torch.Tensor): Precomputed DM probe Influence (M, N_probe)
        Params:
            M (int): Total Degree of Freedom = 2 * N_pixel * N_channels
            N (int): Number of DM Actuators
            N_probe (int): Number of probe pairs
    """

    def __init__(self, data_type, system: SystemSpec):
        super().__init__('PWP', data_type)

        self.M = system.dof
        self.N = system.n_actuators
        self.N_probe = system.n_probes
        self.N_pix_channels = self.M // 2

        self.FLOPs = (
            self.N_pix_channels * self.N_probe * 2 + 
            20 * self.N_pix_channels * self.N_probe
        )
        
        self.mem_access = (4 if self.datatype == 'fp32' else 8) * (
            self.N_pix_channels * self.N_probe * 3 + 
            self.N_probe * 3 * self.N_pix_channels
        )

        self.mem_capacity = (4 if self.datatype == 'fp32' else 8) * (
            self.N_pix_channels * (self.N_probe * 2 + 1)
        )


    def run(self, I_plus, I_minus, delta_p):
        """
        Per-pixel Pairwise Probing Kernel
        
        Args:
            I_plus  (torch.Tensor): Positive Probe Images (N_pixels * N_channels, N_probe)
            I_minus (torch.Tensor): Negative Probe Images (N_pixels * N_channels, N_probe)
            J       (torch.Tensor): Jacobian matrix of shape (M, N)
            probe_u (torch.Tensor): Probe commands of shape (N_probe, N)
        """

        x2_diff_p = (I_plus - I_minus)  # (N_pixels * N_channels, N_probes)
        # Elements Accessed: N_pixels * N_channels * N_probes * 3
        # FLOPs: N_pixels * N_channels * N_probes * 2

        delta_p_Re = delta_p[:self.M//2, :]
        delta_p_Im = delta_p[self.M//2:, :]

        A = torch.stack([-delta_p_Im, delta_p_Re,], dim=2)  # (N_pixel * N_channels, N_probes, 2)

        b = x2_diff_p[..., None]  
        E = torch.linalg.lstsq(A, b).solution.squeeze(-1)  # (N_pixel * N_channels, 2)
        # Elements Accessed: (3 * N_probes) * N_pixels * N_channels
        # FLOPs = 20 * N_pixels * N_channels * N_probes

        return E
    
    def setup(self, device):
        I_plus  = rand_tensor((self.M//2, self.N_probe), self.datatype, device, name="I_plus")
        I_minus = rand_tensor((self.M//2, self.N_probe), self.datatype, device, name="I_minus")
        delta_p = rand_tensor((self.M, self.N_probe), self.datatype, device, name="delta_p")
        return I_plus, I_minus, delta_p