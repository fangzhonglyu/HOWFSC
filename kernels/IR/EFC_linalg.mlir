#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<12656x1xf32>) -> tensor<3619x1xf32> {
    %cst = arith.constant dense_resource<torch_tensor_3619_12656_torch.float32> : tensor<3619x12656xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<3619x1xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<3619x1xf32>) -> tensor<3619x1xf32>
    %2 = linalg.matmul ins(%cst, %arg0 : tensor<3619x12656xf32>, tensor<12656x1xf32>) outs(%1 : tensor<3619x1xf32>) -> tensor<3619x1xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<3619x1xf32>) outs(%0 : tensor<3619x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4 = arith.negf %in : f32
      linalg.yield %4 : f32
    } -> tensor<3619x1xf32>
    return %3 : tensor<3619x1xf32>
  }
}

{-#

#-}
