from setuptools import setup

exec(open('version.py').read())

setup(name='tree_env', version=__version__, install_requires=['gym'])