"""Build Metal Flash Attention PyTorch extension."""
import os
import copy
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

# All .cpp source files (compiled as C++17)
cpp_sources = [
    'csrc/mfa/Metal.cpp',
    'csrc/mfa/ccv_nnc_mfa.cpp',
    'csrc/mfa/ccv_nnc_mfa_error.cpp',
    'csrc/mfa/ccv_nnc_mfa_attention.cpp',
    'csrc/mfa/ccv_nnc_mfa_cast.cpp',
    'csrc/mfa/3rdparty/metal-cpp/Dispatch.cpp',
    'csrc/mfa/v2/AttentionDescriptor.cpp',
    'csrc/mfa/v2/AttentionKernel.cpp',
    'csrc/mfa/v2/AttentionKernel+Precompiled.cpp',
    'csrc/mfa/v2/AttentionKernelDescriptor.cpp',
    'csrc/mfa/v2/NAAttentionDescriptor.cpp',
    'csrc/mfa/v2/NAAttentionKernel.cpp',
    'csrc/mfa/v2/NAAttentionKernelDescriptor.cpp',
    'csrc/mfa/v2/NAMatMulDescriptor.cpp',
    'csrc/mfa/v2/NAMatMulKernel.cpp',
    'csrc/mfa/v2/NAMatMulKernelDescriptor.cpp',
    'csrc/mfa/v2/CodeWriter.cpp',
    'csrc/mfa/v2/GEMMHeaders.cpp',
    'csrc/mfa/v2/CastDescriptor.cpp',
    'csrc/mfa/v2/CastKernel.cpp',
]

# .mm files (Objective-C++)
mm_sources = [
    'csrc/mfa_bridge.mm',
]

# Custom BuildExtension to handle .mm files properly
class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        original_compile = self.compiler._compile

        def patched_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith('.mm'):
                # For .mm files: compile as Objective-C++ with ARC
                postargs = [a for a in extra_postargs]
                postargs.extend(['-fobjc-arc'])
                # Force Objective-C++ mode via cc_args (placed before input file)
                new_cc_args = list(cc_args) + ['-xobjective-c++']
                original_compile(obj, src, ext, new_cc_args, postargs, pp_opts)
            else:
                original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = patched_compile
        super().build_extensions()


setup(
    name='metal-flash-sdpa',
    version='0.1.0',
    ext_modules=[CppExtension(
        name='metal_flash_sdpa._C',
        sources=cpp_sources + mm_sources,
        include_dirs=[
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csrc'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csrc', 'mfa'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csrc', 'mfa', '3rdparty', 'metal-cpp'),
        ],
        extra_compile_args={
            'cxx': ['-std=c++17', '-O2'],
        },
        extra_link_args=[
            '-framework', 'Metal',
            '-framework', 'Foundation',
            '-framework', 'QuartzCore',
        ],
    )],
    cmdclass={'build_ext': CustomBuildExtension},
    packages=['metal_flash_sdpa'],
)
