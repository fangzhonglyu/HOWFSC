from kernels.kernel import Kernel, rand_tensor
from specs.system_spec.system_spec import SystemSpec
import torch

KFF = 35

class EKF(Kernel):
    """
        Kernel for single pixel EKF

        Args:
            J (torch.Tensor): Jacobian matrix of shape (M, N)
            alpha (float): Regularization parameter
        Params:
            M (int): Total Degree of Freedom = 2 * N_pixel * N_channels
            N (int): Number of DM Actuators 
    """
    


    def __init__(self, data_type, system: SystemSpec):
        super().__init__('EKF', data_type)



        self.M = system.dof
        self.N = system.n_actuators

        self.FLOPs = KFF * self.M + 2 * self.M * self.N
        print("EKF FLOPs:", self.FLOPs)
        self.mem_access = (4 if self.datatype == 'fp32' else 8) * (self.M * KFF + self.M * self.N)
        print("EKF mem_access:", self.mem_access)
        self.mem_capacity = (4 if self.datatype == 'fp32' else 8) * (self.M * 7 + self.M)

    def run(self, J, alpha):
        JTJ = torch.matmul(J.T, J)
        JTJ.diagonal().add_(alpha)
        return torch.linalg.solve(JTJ, J.T)
    
    def setup(self, device):
        J = rand_tensor((self.M, self.N), self.datatype, device, name="J")
        alpha = 0.1
        return J, alpha