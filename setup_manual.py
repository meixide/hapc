"""Manual setup using pybind11 directly without CMake."""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import os
from pathlib import Path

class get_pybind_include:
    def __str__(self):
        return pybind11.get_include()

ext_modules = [
    Extension(
        'hapc_core',
        [
            'src/bindings.cpp',
            'src/pchal_design.cpp',
            'src/ridge_wrappers.cpp',
            'src/mkernel.cpp',
            'src/cross_kernel.cpp',
            'src/pcghal_call.cpp',
            'src/pcghal_classi_call.cpp',
            'src/fast_pchal.cpp',
        ],
        include_dirs=[
            get_pybind_include(),
            str(Path(__file__).parent / 'src'),
            '/opt/homebrew/include',  # For Eigen on macOS
            '/usr/include/eigen3',     # For Eigen on Linux
        ],
        library_dirs=[
            '/opt/homebrew/lib',
        ],
        language='c++',
        extra_compile_args=['-std=c++17'],
    ),
]

setup(
    name='hapc',
    version='0.1.0',
    ext_modules=ext_modules,
    packages=['hapc'],
    package_dir={'hapc': 'python/hapc'},
    cmdclass={'build_ext': build_ext},
    install_requires=[
        'numpy>=1.24,<2.3',
        'scipy>=1.7',
        'scikit-learn>=0.24',
    ],
    zip_safe=False,
)
