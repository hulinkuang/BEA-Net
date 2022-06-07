from setuptools import setup, find_namespace_packages

setup(name='BEA',
      packages=find_namespace_packages(include=["BEA", "BEA.*"]),
      version='0.0.1',

      entry_points={
            'console_scripts': [
                  'BEA_train=BEA.run.run_training:main'],
      }
      )
