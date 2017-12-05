from __future__ import print_function

import sys

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readline() if l]

try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)

setup(name='piper-learn',
      version='0.0.1',
      description='A pipeline for machine learning',
      author='Jia Xu',
      package=find_packages(),
      install_required=INSTALL_REQUIRES,
      author_email='dolremi@gmail.com')
