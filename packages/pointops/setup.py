from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

SOURCES = [
    "src/pointops_api.cpp",
    "src/knnquery/knnquery_cuda.cpp",
    "src/knnquery/knnquery_cuda_kernel.cu",
    "src/sampling/sampling_cuda.cpp",
    "src/sampling/sampling_cuda_kernel.cu",
]

setup(
    ext_modules=[
        CUDAExtension(
            "pointops_cuda",
            SOURCES,
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
