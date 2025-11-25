from kernels.efc_v2 import EFC
from specs.compute_spec.compute_specs import ComputeSpec
from specs.system_spec.system_spec import SystemSpec
import time
import torch

k = EFC('fp32', SystemSpec('specs/system_spec/WFIRST.yml'))

k.get_mlir()
