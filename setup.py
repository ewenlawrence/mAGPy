from setuptools import find_packages, setup

# Update when actually setting up in 
setup(
    name='qu-magpy',
    packages=find_packages(include=['qu-magpy']),
    version='0.1.0',
    description='A python 3 library for computing the Adiabatic Gauge Potential',
    author='Ewen Lawrence',
    install_requires=['numpy>=1.26.3'],
    tests_require=[],
)