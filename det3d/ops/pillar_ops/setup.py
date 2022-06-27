from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import glob

src_files = glob.glob('src/*.cpp') + glob.glob('src/*.cu')
print(src_files)

setup(
    name='pillar_cuda_ops',
    ext_modules=[
        CUDAExtension(
            name='pillar_cuda',
            sources=src_files,
            # include_dirs=[' '],
            # libraries=['cuhash'],
            # library_dirs=['point_sample/cuhash/'],
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})

# 'src/hash_ops.cuh',
