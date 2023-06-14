from setuptools import setup, find_packages

setup(
    name='tinynn-py',
    version='0.1.0',
    packages=find_packages(),
    author='Praneeth Vasarla',
    description='A tiny neural network library.',
    install_requires=[
        "numpy"
    ],
)
