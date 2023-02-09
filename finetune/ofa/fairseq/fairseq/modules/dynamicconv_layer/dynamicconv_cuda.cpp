/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor>
dynamicconv_cuda_forward(at::Tensor input, at::Tensor filters, int padding_l);

std::vector<at::Tensor> dynamicconv_cuda_backward(
    at::Tensor gradOutput,
    int padding_l,
    at::Tensor input,
    at::Tensor filters);

#define CHECK_CUDA(x) \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<at::Tensor>
dynamicconv_forward(at::Tensor input, at::Tensor filters, int padding_l) {
  CHECK_INPUT(input);
  CHECK_INPUT(filters);

  return dynamicconv_cuda_forward(input, filters, padding_l);
}

std::vector<at::Tensor> dynamicconv_backward(
    at::Tensor gradOutput,
    int padding_l,
    at::Tensor input,
    at::Tensor filters) {
  CHECK_INPUT(gradOutput);
  CHECK_INPUT(input);
  CHECK_INPUT(filters);

  return dynamicconv_cuda_backward(gradOutput, padding_l, input, filters);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dynamicconv_forward, "dynamicconv forward (CUDA)");
  m.def("backward", &dynamicconv_backward, "dynamicconv backward (CUDA)");
}
