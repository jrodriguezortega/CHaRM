from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

SOURCES = [
    "src/pointnet2_api.cpp",
    "src/ball_query.cpp",
    "src/ball_query_gpu.cu",
    "src/group_points.cpp",
    "src/group_points_gpu.cu",
    "src/interpolate.cpp",
    "src/interpolate_gpu.cu",
    "src/sampling.cpp",
    "src/sampling_gpu.cu",
]

setup(
    ext_modules=[
        CUDAExtension(
            "pointnet2_batch_cuda",
            SOURCES,
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
