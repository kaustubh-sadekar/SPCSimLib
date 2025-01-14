from setuptools import setup

setup(
    name='SPCSim',
    version='1.0.0',    
    description='Python package for single photon cameras',
    author='Kaustubh Sadekar, Atul Ingle',
    packages=['SPCSim', 'SPCSim.data_loaders','SPCSim.postproc','SPCSim.sensors', 'SPCSim.utils'],

    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.9',
    ],
)