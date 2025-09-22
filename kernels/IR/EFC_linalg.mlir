module {
  func.func @main(%arg0: tensor<3619x12656xf32>, %arg1: tensor<12656x1xf32>) -> tensor<3619x1xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<3619x1xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<3619x1xf32>) -> tensor<3619x1xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<3619x12656xf32>, tensor<12656x1xf32>) outs(%1 : tensor<3619x1xf32>) -> tensor<3619x1xf32>
    return %2 : tensor<3619x1xf32>
  }
}
