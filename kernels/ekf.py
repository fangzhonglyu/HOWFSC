from kernels.kernel import Kernel, rand_tensor
from specs.system_spec.system_spec import SystemSpec
import torch

KFF = 35

class EKF(Kernel):
    """
    Extended Kalman Filter for High-Order Wavefront Sensing and Control (HOWFSC).

    Implements the EKF algorithm from Pogorelyuk & Kasdin (2019) Section 4.1 and Appendix A.

    State per pixel: [Re{E^OL_ij}, Im{E^OL_ij}, I_ij]
    - E^OL_ij: Open-loop electric field at pixel (i,j)
    - I_ij: Incoherent intensity at pixel (i,j)

    Core computation: Kalman gain calculation and state/covariance updates
    K = P*H^T / (H*P*H^T + R)  (Eq. 39)
    x+ = x + K*(y - y_pred)    (Eq. 37)
    P+ = (I - K*H)*P + Q       (Eq. 41)

    Args:
        data_type (str): 'fp32' or 'fp64'
        system (SystemSpec): System specification

    Params:
        M (int): Number of pixels in dark hole
        state_dim (int): State dimension per pixel (3)
    """

    def __init__(self, data_type, system: SystemSpec):
        super().__init__('EKF', data_type)

        # Number of pixels = DOF / 2 (real and imaginary parts)
        self.M = system.dof // 2
        self.state_dim = 3  # [Re{E}, Im{E}, I] per pixel

        # FLOPs per pixel EKF update:
        # - H*P: 3x3 matrix-vector = 9 ops
        # - H*P*H^T: 3 ops (dot product)
        # - K = P*H^T/S: 3x3 + 3 = 12 ops
        # - Innovation: 3 ops
        # - State update: 3 ops
        # - (I-K*H): 3x3 = 9 ops
        # - P update: 3x3 matrix mult + add = 18 ops
        # Total ≈ 57 ops, but using KFF=35 as baseline
        self.FLOPs = KFF * self.M
        print(f"EKF FLOPs: {self.FLOPs} (for {self.M} pixels)")

        bytes_per_elem = 4 if self.datatype == 'fp32' else 8
        # Memory access: state(3) + P(9) + H(3) + intermediate results
        self.mem_access = bytes_per_elem * self.M * (3 + 9 + 3 + 6)
        print(f"EKF mem_access: {self.mem_access} bytes")

        # Memory capacity: state(3) + P(9) per pixel
        self.mem_capacity = bytes_per_elem * self.M * (3 + 9)

    def run(self, state, P, H, R, innovation, Q):
        """
        Core EKF update computation for profiling (Eqs. 37-41).

        Args:
            state: State vector [3] or [N, 3]
            P: Covariance matrix [3, 3] or [N, 3, 3]
            H: Measurement Jacobian [3] or [N, 3]
            R: Measurement noise variance (scalar or [N])
            innovation: y - y_pred (scalar or [N])
            Q: Process noise covariance [3, 3] or [N, 3, 3]

        Returns:
            state_new: Updated state
            P_new: Updated covariance
        """
        # Kalman gain: K = P*H^T / (H*P*H^T + R)  (Eq. 39)
        if H.dim() == 1:
            # Single pixel case
            PH = torch.matmul(P, H)  # [3]
            HPH = torch.dot(H, PH)    # scalar
            S = HPH + R
            K = PH / S.clamp(min=1e-10)  # [3]

            # State update (Eq. 37)
            state_new = state + K * innovation

            # Covariance update (Eq. 41)
            I_KH = torch.eye(3, dtype=P.dtype, device=P.device) - torch.outer(K, H)
            P_new = torch.matmul(I_KH, P) + Q
        else:
            # Batched case [N, 3]
            PH = torch.einsum('nij,nj->ni', P, H)  # [N, 3]
            HPH = torch.einsum('ni,ni->n', H, PH)   # [N]
            S = HPH + R
            K = PH / S.clamp(min=1e-10).unsqueeze(-1)  # [N, 3]

            # State update
            state_new = state + K * innovation.unsqueeze(-1)

            # Covariance update
            I_mat = torch.eye(3, dtype=P.dtype, device=P.device).unsqueeze(0)  # [1, 3, 3]
            KH = torch.einsum('ni,nj->nij', K, H)  # [N, 3, 3]
            I_KH = I_mat - KH
            P_new = torch.einsum('nij,njk->nik', I_KH, P) + Q

        return state_new, P_new

    def setup(self, device):
        """Setup test inputs for profiling the core EKF computation."""
        # Use batched format for efficiency [N_pixels, state_dim]
        N = self.M

        # State vector per pixel [N, 3]
        state = rand_tensor((N, 3), self.datatype, device, name="state") * 0.1

        # Covariance matrix per pixel [N, 3, 3]
        P = torch.eye(3, dtype=torch.float32 if self.datatype == 'fp32' else torch.float64,
                     device=device).unsqueeze(0).repeat(N, 1, 1) * 0.1

        # Measurement Jacobian H = [2*Re{E}, 2*Im{E}, 1] per pixel [N, 3]
        H = rand_tensor((N, 3), self.datatype, device, name="H")
        H[:, 2] = 1.0  # Last element is always 1

        # Measurement noise variance [N]
        R = rand_tensor((N,), self.datatype, device, name="R") * 0.5 + 0.5

        # Innovation (y - y_pred) [N]
        innovation = rand_tensor((N,), self.datatype, device, name="innovation") * 2.0 - 1.0

        # Process noise covariance [N, 3, 3]
        Q = torch.zeros(N, 3, 3, dtype=torch.float32 if self.datatype == 'fp32' else torch.float64,
                       device=device)
        Q[:, 0:2, 0:2] = 0.01  # Electric field drift
        Q[:, 2, 2] = 0.001     # Intensity drift

        return state, P, H, R, innovation, Q