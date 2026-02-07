// Vendored from https://github.com/liuliu/ccv (unstable branch, lib/nnc/mfa/)
// Copyright (c) 2010 Liu Liu. BSD 3-clause license — see THIRD_PARTY_LICENSES.
// Local modifications: none

#ifndef CastKernel_hpp
#define CastKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>
#include "CastDescriptor.hpp"

struct CastKernel {
  NS::SharedPtr<MTL::Library> library;
  
  std::string source;

  unsigned short threadgroupMemoryAllocation;

  /// The number of threads per group.
  MTL::Size threadgroupSize;

  uint8_t value;

  GEMMOperandPrecision fromMemoryPrecision;

  GEMMOperandPrecision memoryPrecision;

  CastKernel(CastKernelDescriptor descriptor, MTL::Device *const device);

private:
  unsigned short createThreadgroupMemoryAllocation() const noexcept;
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif /* CastKernel_hpp */

