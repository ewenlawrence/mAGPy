from setuptools import find_packages, setup

from magpy.version import __version__

import os
import sys

# Update path with current directory
curr_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, curr_dir)

# set the descriptions
short_desc = "A python 3 library for computing the Adiabatic Gauge Potential"
with open(os.path.join(curr_dir, "README.md"), "r") as f:
    long_description = f.read()

# load requirements list
with open(os.path.join(curr_dir, "requirements.txt"), "r") as requirements_file:
    requirements = requirements_file.readlines()

#TODO add keywords
setup(name='qu-magpy',
      version=__version__,
      description=short_desc,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Ewen Lawrence',
      author_email='ewenlawrence@gmail.com',
      url='https://github.com/ewenlawrence/mAGPy',
      license="Apache 2.0",
      packages=find_packages(include=['magpy']),
      install_requires=requirements,
      python_requires=('>=3.9.0'))
