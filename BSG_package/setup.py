from setuptools import setup, find_namespace_packages

setup(name='BSG',
      packages=find_namespace_packages(include=["BSG", "BSG.*"]),
      version='0.0.1'
      )
