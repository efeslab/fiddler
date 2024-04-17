import torch.utils.cpp_extension as torch_cpp_ext
from setuptools import setup

# include_dirs = cpp_extension.include_paths()
# library_dirs = cpp_extension.library_paths()
extra_compile_args = ["-fopenmp", "-mavx512bf16", "-O3"]  # OpenMP and AVX512bf16 flags

ext_modules = [
    torch_cpp_ext.CppExtension(
        "bfloat16_expert",
        ["bfloat16-expert.cpp"],
        # include_dirs=include_dirs,
        # library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=["-lgomp"],  # Linking against the GNU OpenMP library
    ),
]

setup(
    name="bfloat16-expert",
    version="0.1",
    author="Your Name",
    description="A Python package with C++ extension using PyTorch, OpenMP, and AVX512",
    ext_modules=ext_modules,
    cmdclass={"build_ext": torch_cpp_ext.BuildExtension},
    zip_safe=False,
)
