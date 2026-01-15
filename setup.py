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
        import shutil
        import time
        
        # Search recursively for the built library
        # Try multiple patterns to ensure we find it on all platforms
        search_patterns = [
            os.path.join(str(build_temp), "**", "hapc_core.*"),
            os.path.join(str(build_temp), "**", "*", "hapc_core.*"),
            os.path.join(str(build_temp), "**", "*", "*", "hapc_core.*"),
        ]
        
        found = False
        for pattern in search_patterns:
            for lib in glob.glob(pattern, recursive=True):
                if lib.endswith(('.pyd', '.so', '.dylib')):
                    try:
                        dest = lib_dir / Path(lib).name
                        # Retry on Windows if file is locked
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                shutil.copy2(lib, dest)
                                print(f"[OK] Copied {lib} to {dest}")
                                found = True
                                break
                            except (OSError, PermissionError) as e:
                                if attempt < max_retries - 1:
                                    time.sleep(0.5)
                                else:
                                    raise
                        if found:
                            break
                    except Exception as e:
                        print(f"Warning: Failed to copy {lib}: {e}")
            if found:
                break
        
        if not found:
            print(f"ERROR: No compiled library found in build directory {build_temp}")
            print(f"  Searched for: hapc_core.pyd (Windows), hapc_core.so (Linux), hapc_core.dylib (macOS)")
            raise RuntimeError("Failed to locate compiled hapc_core extension")
        
        # Don't call parent run() to avoid setuptools trying to clean Windows locked files
        # Just mark as complete
        self.build_libs = []

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
    # Correctly name the extension to be part of the 'hapc' package
    ext_modules=[Extension("hapc.hapc_core", [])],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
    include_package_data=True,
)