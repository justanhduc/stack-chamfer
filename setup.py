from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import glob

ext_root = os.getcwd()
ext_src = glob.glob(os.path.join(ext_root, 'csrc/*.cpp')) + glob.glob(os.path.join(ext_root, 'csrc/*.cu'))
ext_include = os.path.join(ext_root, 'include')
setup(
    name='stack-chamfer',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('stack_chamfer._cuda',
                      sources=ext_src,
                      include_dirs=[ext_include]
                      )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
