"""Setup for HAPC package (root level for remote installation)."""

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys
import os
from pathlib import Path

class CMakeBuild(build_ext):
    """Custom build using CMake."""
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake is required to build hapc. Install with: pip install cmake")
        
        build_temp = Path(self.build_temp) / "cmake_build"
        build_temp.mkdir(parents=True, exist_ok=True)
        
        project_root = Path(__file__).parent
        
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]
        
        subprocess.check_call(
            ["cmake", str(project_root)] + cmake_args,
            cwd=str(build_temp)
        )
        
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release"],
            cwd=str(build_temp)
        )
        
        # Copy library to package
        lib_dir = project_root / "python" / "hapc"
        lib_dir.mkdir(parents=True, exist_ok=True)
        
        import glob
        for lib in glob.glob(os.path.join(str(build_temp), "hapc_core.*")):
            import shutil
            dest = lib_dir / Path(lib).name
            shutil.copy2(lib, dest)
            print(f"Copied {lib} to {dest}")

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
    description="Hierarchical Additive Polynomial Complexity regression",
    long_description=readme_content,
    long_description_content_type="text/markdown",
    author="Carlos García Meixide",
    author_email="cgmeixide@gmail.com",
    url="https://github.com/meixide/hapc",
    license="MIT",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[Extension("hapc._core", [])],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
    include_package_data=True,
)