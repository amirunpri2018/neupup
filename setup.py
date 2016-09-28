from setuptools import setup
from setuptools import find_packages

install_requires = [
    'numpy',
]

setup(name='neupup',
      version='0.1.0',
      description='Perform neural puppet transformations on images',
      author='Tom White',
      author_email='tom@sixdozen.com',
      url='https://github.com/dribnet/neupup',
      download_url='https://github.com/dribnet/neupup/tarball/0.1.0',
      license='MIT',
      entry_points={
          'console_scripts': ['neupup = neupup.neupup:main']
      },
      install_requires=install_requires,
      packages=find_packages())
