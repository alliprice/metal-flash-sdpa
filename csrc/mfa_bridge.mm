// mfa_bridge.mm — Objective-C++ bridge between PyTorch MPS tensors and Metal Flash Attention
#include <torch/extension.h>
#include <torch/mps.h>
#include <ATen/mps/MPSDevice.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "mfa/ccv_nnc_mfa.hpp"
#include "mfa/ccv_nnc_mfa_attention.hpp"

// Extract the underlying MTLBuffer from a PyTorch MPS tensor's storage.
// This is the zero-copy path: PyTorch MPS tensors store data in MTLBuffers.
static inline id<MTLBuffer> getMTLBufferStorage(const at::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

static inline size_t getMTLBufferOffset(const at::Tensor& tensor) {
  // storage_offset is in elements, multiply by element size
  return tensor.storage_offset() * tensor.element_size();
}

// ============================================================================
// Global MFA context (lazily initialized, reused across calls)
// ============================================================================
static ccv_nnc_mfa_context_t* g_mfa_ctx = nullptr;

static ccv_nnc_mfa_context_t* get_mfa_context() {
  if (!g_mfa_ctx) {
    // Use PyTorch's MPS device to avoid device mismatch
    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    auto* cpp_device = (__bridge MTL::Device*)device;
    g_mfa_ctx = ccv_nnc_init_mfa_context(cpp_device);
  }
  return g_mfa_ctx;
}

// ============================================================================
// Map PyTorch dtype to MTL::DataType
// ============================================================================
static uint64_t torchDtypeToMTL(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat: return MTL::DataTypeFloat;
    case at::kHalf: return MTL::DataTypeHalf;
    case at::kBFloat16: return MTL::DataTypeBFloat;
    default:
      TORCH_CHECK(false, "Unsupported dtype for MFA attention: ", dtype);
  }
}

// ============================================================================
// Shared helpers
// ============================================================================
static void validate_attention_inputs(
    const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V) {
  TORCH_CHECK(Q.device().is_mps(), "Q must be on MPS device");
  TORCH_CHECK(K.device().is_mps(), "K must be on MPS device");
  TORCH_CHECK(V.device().is_mps(), "V must be on MPS device");
  TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
  TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
  TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
  TORCH_CHECK(Q.dim() == 4, "Q must be 4D [B,R,Hq,D]");
  TORCH_CHECK(K.dim() == 4, "K must be 4D [B,C,Hk,D]");
  TORCH_CHECK(V.dim() == 4, "V must be 4D [B,C,Hk,D]");
}

static ccv_nnc_mfa_attention_params_t make_attention_params(
    int64_t B, int64_t R, int64_t C, int64_t Hq, int64_t Hk, int64_t D,
    float scale, uint8_t type, at::ScalarType dtype) {
  ccv_nnc_mfa_attention_params_t params = {};
  params.type = type;
  params.R = (uint32_t)R;
  params.C = (uint32_t)C;
  params.Hq = (uint32_t)Hq;
  params.Hk = (uint32_t)Hk;
  params.D = (uint32_t)D;
  params.Q_trans = 0;
  params.K_trans = 0;
  params.V_trans = 0;
  params.O_trans = 0;
  params.alpha = scale;
  params.batched = (B > 1) ? 1 : 0;
  params.masked = 0;
  params.upcast = 1;
  params.use_neural_accelerators = 0;
  params.data_type = torchDtypeToMTL(dtype);
  memset(params.batch_dims_q, 0, sizeof(params.batch_dims_q));
  memset(params.batch_dims_mask, 0, sizeof(params.batch_dims_mask));
  if (B > 1) {
    params.batch_dims_q[0] = (uint32_t)B;
  }
  return params;
}

// ============================================================================
// Forward attention: Q[B,R,Hq,D], K[B,C,Hk,D], V[B,C,Hk,D] -> {O[B,R,Hq,D], LSE[B*Hq*R]}
// ============================================================================
std::vector<torch::Tensor> mfa_attention_forward(
    const torch::Tensor& Q,   // [B, R, Hq, D]
    const torch::Tensor& K,   // [B, C, Hk, D]
    const torch::Tensor& V,   // [B, C, Hk, D]
    double scale
) {
  validate_attention_inputs(Q, K, V);

  auto B  = Q.size(0);
  auto R  = Q.size(1);
  auto Hq = Q.size(2);
  auto D  = Q.size(3);
  auto C  = K.size(1);
  auto Hk = K.size(2);

  TORCH_CHECK(K.size(0) == B && V.size(0) == B, "Batch size mismatch");
  TORCH_CHECK(V.size(1) == C, "K and V sequence length mismatch");
  TORCH_CHECK(K.size(3) == D && V.size(3) == D, "Head dimension mismatch");
  TORCH_CHECK(V.size(2) == Hk, "K and V head count mismatch");

  // Allocate outputs
  auto O = torch::empty_like(Q);
  // LSE is always float32, flat layout [B*Hq*R]
  auto LSE = torch::empty({B * Hq * R},
      torch::TensorOptions().device(Q.device()).dtype(torch::kFloat32));

  auto params = make_attention_params(B, R, C, Hq, Hk, D, (float)scale, 0, Q.scalar_type());

  auto* ctx = get_mfa_context();
  ccv_nnc_mfa_prepare_attention(ctx, params);

  @autoreleasepool {
    dispatch_sync(torch::mps::get_dispatch_queue(), ^{
      @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        auto* cppEncoder = (__bridge MTL::ComputeCommandEncoder*)encoder;
        auto* cppCmdBuf = (__bridge MTL::CommandBuffer*)cmdBuf;
        MTL::CommandBatch batch(cppCmdBuf, cppEncoder);

        // Forward: tensors[0]=Q, [1]=K, [2]=V, [3]=O, [4]=mask(null), [5]=LSE
        MTL::Buffer* tensors[6];
        tensors[0] = (__bridge MTL::Buffer*)getMTLBufferStorage(Q);
        tensors[1] = (__bridge MTL::Buffer*)getMTLBufferStorage(K);
        tensors[2] = (__bridge MTL::Buffer*)getMTLBufferStorage(V);
        tensors[3] = (__bridge MTL::Buffer*)getMTLBufferStorage(O);
        tensors[4] = nullptr;
        tensors[5] = (__bridge MTL::Buffer*)getMTLBufferStorage(LSE);

        size_t tensor_offsets[6];
        tensor_offsets[0] = getMTLBufferOffset(Q);
        tensor_offsets[1] = getMTLBufferOffset(K);
        tensor_offsets[2] = getMTLBufferOffset(V);
        tensor_offsets[3] = getMTLBufferOffset(O);
        tensor_offsets[4] = 0;
        tensor_offsets[5] = getMTLBufferOffset(LSE);

        ccv_nnc_mfa_encode_attention(ctx, params, &batch, tensors, tensor_offsets);

        [encoder endEncoding];
      }
    });
  }

  torch::mps::commit();

  return {O, LSE};
}

// ============================================================================
// Backward attention: computes dQ, dK, dV
// ============================================================================
std::vector<torch::Tensor> mfa_attention_backward(
    const torch::Tensor& Q,    // [B, R, Hq, D]
    const torch::Tensor& K,    // [B, C, Hk, D]
    const torch::Tensor& V,    // [B, C, Hk, D]
    const torch::Tensor& O,    // [B, R, Hq, D]
    const torch::Tensor& LSE,  // [B*Hq*R] float32
    const torch::Tensor& dO,   // [B, R, Hq, D]
    double scale
) {
  validate_attention_inputs(Q, K, V);
  TORCH_CHECK(O.device().is_mps() && O.is_contiguous(), "O must be contiguous MPS tensor");
  TORCH_CHECK(LSE.device().is_mps() && LSE.is_contiguous(), "LSE must be contiguous MPS tensor");
  TORCH_CHECK(dO.device().is_mps() && dO.is_contiguous(), "dO must be contiguous MPS tensor");

  auto B  = Q.size(0);
  auto R  = Q.size(1);
  auto Hq = Q.size(2);
  auto D  = Q.size(3);
  auto C  = K.size(1);
  auto Hk = K.size(2);

  TORCH_CHECK(K.size(0) == B && V.size(0) == B, "Batch size mismatch");
  TORCH_CHECK(V.size(1) == C, "K and V sequence length mismatch");
  TORCH_CHECK(K.size(3) == D && V.size(3) == D, "Head dimension mismatch");
  TORCH_CHECK(V.size(2) == Hk, "K and V head count mismatch");

  // MFA backward uses Hq for dK/dV batch strides (not Hk), so allocate with Hq
  auto dQ = torch::empty_like(Q);                                       // [B, R, Hq, D]
  auto dK = torch::empty({B, C, Hq, D}, K.options());                   // [B, C, Hq, D]
  auto dV = torch::empty({B, C, Hq, D}, V.options());                   // [B, C, Hq, D]

  auto params = make_attention_params(B, R, C, Hq, Hk, D, (float)scale, 1, Q.scalar_type());

  auto* ctx = get_mfa_context();
  ccv_nnc_mfa_prepare_attention(ctx, params);

  @autoreleasepool {
    dispatch_sync(torch::mps::get_dispatch_queue(), ^{
      @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        auto* cppEncoder = (__bridge MTL::ComputeCommandEncoder*)encoder;
        auto* cppCmdBuf = (__bridge MTL::CommandBuffer*)cmdBuf;
        MTL::CommandBatch batch(cppCmdBuf, cppEncoder);

        // Backward: [0]=Q [1]=K [2]=V [3]=O [4]=LSE [5]=dO [6]=dQ [7]=dK [8]=dV
        MTL::Buffer* tensors[9];
        tensors[0] = (__bridge MTL::Buffer*)getMTLBufferStorage(Q);
        tensors[1] = (__bridge MTL::Buffer*)getMTLBufferStorage(K);
        tensors[2] = (__bridge MTL::Buffer*)getMTLBufferStorage(V);
        tensors[3] = (__bridge MTL::Buffer*)getMTLBufferStorage(O);
        tensors[4] = (__bridge MTL::Buffer*)getMTLBufferStorage(LSE);
        tensors[5] = (__bridge MTL::Buffer*)getMTLBufferStorage(dO);
        tensors[6] = (__bridge MTL::Buffer*)getMTLBufferStorage(dQ);
        tensors[7] = (__bridge MTL::Buffer*)getMTLBufferStorage(dK);
        tensors[8] = (__bridge MTL::Buffer*)getMTLBufferStorage(dV);

        size_t tensor_offsets[9];
        tensor_offsets[0] = getMTLBufferOffset(Q);
        tensor_offsets[1] = getMTLBufferOffset(K);
        tensor_offsets[2] = getMTLBufferOffset(V);
        tensor_offsets[3] = getMTLBufferOffset(O);
        tensor_offsets[4] = getMTLBufferOffset(LSE);
        tensor_offsets[5] = getMTLBufferOffset(dO);
        tensor_offsets[6] = getMTLBufferOffset(dQ);
        tensor_offsets[7] = getMTLBufferOffset(dK);
        tensor_offsets[8] = getMTLBufferOffset(dV);

        ccv_nnc_mfa_encode_attention(ctx, params, &batch, tensors, tensor_offsets);

        [encoder endEncoding];
      }
    });
  }

  torch::mps::commit();

  return {dQ, dK, dV};
}

// ============================================================================
// pybind11 module
// ============================================================================
PYBIND11_MODULE(_C, m) {
  m.def("mfa_attention_forward", &mfa_attention_forward,
        "Metal Flash Attention forward pass",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("scale"));
  m.def("mfa_attention_backward", &mfa_attention_backward,
        "Metal Flash Attention backward pass",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"),
        py::arg("LSE"), py::arg("dO"), py::arg("scale"));
}
