// Vendored from https://github.com/liuliu/ccv (unstable branch, lib/nnc/mfa/)
// Copyright (c) 2010 Liu Liu. BSD 3-clause license — see THIRD_PARTY_LICENSES.
// Local modifications: none

#ifndef MFA_PIPELINE_VALUE_HPP_
#define MFA_PIPELINE_VALUE_HPP_

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"

template<typename T>
struct PipelineValue {
  T* kernel;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
  NS::SharedPtr<MTL::IndirectCommandBuffer> indirect1; // This is optional.
  NS::SharedPtr<MTL::Function> function; // This is optional.
  NS::SharedPtr<MTL::ComputePipelineState> second; // This is optional.
  NS::SharedPtr<MTL::IndirectCommandBuffer> indirect2; // This is optional.
};

#endif
