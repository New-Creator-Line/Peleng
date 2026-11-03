import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "peleng_core",
        [
            "src/peleng_core/bindings.cpp",
        ],
        include_dirs=["src/peleng_core"],
        cxx_std=20,
    ),
]

setup(
    name="peleng_core",
    version="1.0.0",
    author="Peleng 3D Team",
    description="Peleng 3D Vision Core C++ Module",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.10",
)
