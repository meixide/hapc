"""Setup for HAPC package (root level for remote installation)."""

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess
import sys
from pathlib import Path

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        # Add parallel build flag only on non-Windows platforms
        # On Windows, MSBuild doesn't support -j flag and handles parallelization automatically
        if sys.platform != 'win32':
            build_args += ['--', '-j2']


        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                          self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

# Try to read version, fallback to default
version = "0.1.0"
init_file = Path(__file__).parent / "python" / "hapc" / "__init__.py"
if init_file.exists():
    with open(init_file, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"').strip("'")
                break

# Try to read README
readme_content = ""
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r") as f:
        readme_content = f.read()

setup(
    name="hapc",
    version=version,
    description="Highly Adaptive Principal Components",
    long_description=readme_content,
    long_description_content_type="text/markdown",
    author="Carlos García Meixide",
    author_email="cgmeixide@gmail.com",
    url="https://github.com/meixide/hapc",
    license="MIT",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[CMakeExtension('hapc/hapc_core', sourcedir=os.path.dirname(os.path.abspath(__file__)))],
    cmdclass=dict(build_ext=CMakeBuild),
    python_requires=">=3.8",
    # Dependencies are defined in pyproject.toml [project.dependencies]
    # install_requires is omitted here to avoid conflicts with pyproject.toml
    include_package_data=True,
    zip_safe=False,
)