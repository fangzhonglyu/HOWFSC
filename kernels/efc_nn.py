import torch
from kernels.efc_v2 import EFC
from specs.system_spec.system_spec import SystemSpec
from torch_mlir.fx import export_and_import
import os

class KernelModule(torch.nn.Module):
    def __init__(self, kernel_instance):
        super().__init__()
        self.kernel_instance = kernel_instance
        
        self.MJT = torch.nn.Parameter(torch.randn((self.kernel_instance.N, self.kernel_instance.M), dtype=torch.float32 if self.kernel_instance.datatype == 'fp32' else torch.float64))

    def forward(self, E):
        return -torch.matmul(self.MJT, E)
    
efc = EFC('fp32', SystemSpec('specs/system_spec/WFIRST.yml'))
m = KernelModule(efc).eval()


mlir_torch = export_and_import(m, torch.rand((efc.M), dtype=torch.float32), output_type="torch")
file_dir = os.path.dirname(os.path.abspath(__file__))
# torch_mlir_file = os.path.join(file_dir,"IR",f"{efc.name}_torch.mlir")
# with open(torch_mlir_file, 'w') as f:
#     f.write(str(mlir_torch))
# print(f"Saved Torch MLIR to {torch_mlir_file}")
mlir_linalg = export_and_import(m, torch.rand((efc.M), dtype=torch.float32), output_type="linalg-on-tensors")
linalg_mlir_file = os.path.join(file_dir,"IR",f"{efc.name}_linalg.mlir")
with open(linalg_mlir_file, 'w') as f:
    mlir_linalg.operation.print(file=f, large_elements_limit=10)
print(f"Saved Linalg MLIR to {linalg_mlir_file}")
