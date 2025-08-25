from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os, os.path as osp

ROOT = osp.dirname(osp.abspath(__file__))
EIGEN_CONDA = osp.join(os.environ.get("CONDA_PREFIX", ""), "include", "eigen3")

setup(
    name='droid_backends',
    ext_modules=[
        CUDAExtension(
            'droid_backends',
            include_dirs=[
                osp.join(ROOT, 'thirdparty/eigen'),
                EIGEN_CONDA,                     # <--- 关键：conda eigen 头文件
            ],
            sources=[
                'src/droid.cpp',
                'src/droid_kernels.cu',
                'src/correlation_kernels.cu',
                'src/altcorr_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_89,code=sm_89',  
                    # 如需 V100 再加 sm_70；老卡 60/61 可移除
                ]
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)

setup(
    name='lietorch',
    version='0.2',
    description='Lie Groups for PyTorch',
    packages=['lietorch'],
    package_dir={'': 'thirdparty/lietorch'},
    ext_modules=[
        CUDAExtension(
            'lietorch_backends',
            include_dirs=[
                osp.join(ROOT, 'thirdparty/lietorch/lietorch/include'),
                osp.join(ROOT, 'thirdparty/eigen'),
                EIGEN_CONDA,                     # <--- 关键
            ],
            sources=[
                'thirdparty/lietorch/lietorch/src/lietorch.cpp',
                'thirdparty/lietorch/lietorch/src/lietorch_gpu.cu',
                'thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_89,code=sm_89',  
                ]
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
