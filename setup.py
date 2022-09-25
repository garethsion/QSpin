import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name="Quantum Spin",
    ext_modules=cythonize('c_lanczos.pyx'),
    include_dirs=[np.get_include()],
    zip_safe=False
)