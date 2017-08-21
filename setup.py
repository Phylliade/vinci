from setuptools import setup
from setuptools import find_packages

setup(
    name='vinci',
    version='1.1.0',
    description='Deep Reinforcement Learning framework',
    author='Pierre Manceron',
    url='https://github.com/phylliade/vinci',
    license='MIT',
    install_requires=['numpy', 'keras>=2.0.0', 'gym>=0.9.2'],
    extras_require={'plot': ['matplotlib', 'seaborn'], 'analytics': ["pandas"]},
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages())
