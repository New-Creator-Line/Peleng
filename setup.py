import os
import sys
from pathlib import Path
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

include_dirs = ["src/peleng_core"]
library_dirs = []
libraries = []
define_macros = []

nvdec_sdk_dir = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\Video_Codec_SDK_13.1.24")
cuda_dir = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"))

if nvdec_sdk_dir.exists() and cuda_dir.exists():
    include_dirs.append(str(nvdec_sdk_dir / "Interface"))
    include_dirs.append(str(cuda_dir / "include"))
    library_dirs.append(str(nvdec_sdk_dir / "Lib" / "win" / "x64"))
    library_dirs.append(str(cuda_dir / "lib" / "x64"))
    libraries.extend(["nvcuvid", "nvencodeapi", "cuda"])
    define_macros.append(("PELENG_HAS_NVDEC_SDK", "1"))

ext_modules = [
    Pybind11Extension(
        "peleng_core",
        [
            "src/peleng_core/bindings.cpp",
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        define_macros=define_macros,
        cxx_std=20,
    ),
]

setup(
    name="peleng_core",
    version="1.1.0",
    author="Peleng 3D Team",
    description="Peleng 3D Vision Core C++ Module with NVIDIA Video Codec SDK NVDEC",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.10",
)
