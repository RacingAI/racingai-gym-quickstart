try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(name='f110_gym',
      version='0.2',
      author='Hongrui Zheng',
      author_email='billyzheng.bz@gmail.com',
      url='https://f1tenth.org',
      install_requires=['gym', 'numpy', 'Pillow', 'scipy', 'numba', 'pyyaml'],
      packages=find_packages()
      )