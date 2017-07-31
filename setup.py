from setuptools import setup
from setuptools import find_packages


setup(name='vinci',
      version='1.0.0~dev',
      description='Deep Reinforcement Learning for Keras',
      author='Matthias Plappert, Pierre Manceron',
      url='https://github.com/phylliade/vinci',
      license='MIT',
      install_requires=['keras>=1.0.7'],
      extras_require={
          'gym': ['gym'],
      },
      packages=find_packages())
