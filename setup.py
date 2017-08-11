from setuptools import setup
from setuptools import find_packages


setup(name='vinci',
      version='1.0.0~dev',
      description='Deep Reinforcement Learning framework',
      author='Pierre Manceron',
      url='https://github.com/phylliade/vinci',
      license='MIT',
      install_requires=['keras>=1.0.7', 'seaborn'],
      extras_require={
          'gym': ['gym'],
      },
      packages=find_packages())
