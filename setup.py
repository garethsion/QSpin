import numpy as np
from distutils.core import setup
from setuptools import find_packages
from Cython.Build import cythonize

setup(
    name="qent",
    version="0.1.0",
    author = "Gareth Sion Jones",
    author_email="gareth.jones@materials.ox.ac.uk",
    ext_modules=cythonize('qent/dmrg/c_lanczos.pyx', build_dir="qent/dmrg/build"),
    packages=find_packages(exclude=['*test']),
    include_dirs=[np.get_include()],
    zip_safe=False
)