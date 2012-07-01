from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(ext_modules=[Extension(
                   name="btfcount",
                   sources=["tfcount.pyx"],
                   language="c",
                   include_dirs = ["/usr/include/bcutils", "/usr/include/btfcount"],
                   library_dirs = ["/usr/lib"],
                   libraries = ["bcutils", "btfcount"],
                   )],
      cmdclass={'build_ext': build_ext})