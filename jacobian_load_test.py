import numpy as np
import torch
import time

jacobian_DM1_npy = np.load('inputs/jacobian_DM1_546nm.npy')
print (f"Jacobian shape: {jacobian_DM1_npy.shape}, size: {jacobian_DM1_npy.nbytes / 1e9} GB, datatype: {jacobian_DM1_npy.dtype}")

jacobian_DM2_npy = np.load('inputs/jacobian_DM2_546nm.npy')
print (f"Jacobian shape: {jacobian_DM2_npy.shape}, size: {jacobian_DM2_npy.nbytes / 1e9} GB, datatype: {jacobian_DM2_npy.dtype}")