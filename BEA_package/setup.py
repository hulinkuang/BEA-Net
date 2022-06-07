from setuptools import setup, find_namespace_packages

setup(name='BEA',
      packages=find_namespace_packages(include=["BEA", "BEA.*"]),
      version='0.0.1'
      )
