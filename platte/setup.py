from setuptools import setup
from importlib.machinery import SourceFileLoader

with open('README.md') as file:
    long_description = file.read()

version = SourceFileLoader('platte.version', 'platte/version.py').load_module()

setup(
   name='platte',
   version=version.version,
   description='A toolbox to accelerate neurological diagnostics.',
   author='Ard Kastrati & Maxim Huber',
   author_email='akastrati@ethz.ch, maximhuber@student.ethz.ch',
   url='',
   packages=['platte'],
   long_description=long_description,
   long_description_content_type='text/markdown',
   keywords='We make neurological diagnostics 10x faster',
   install_requires=[],
)
