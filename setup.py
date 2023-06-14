from setuptools import setup, find_packages

setup(
    name='tinynn-py',
    version='1.0.2',
    packages=find_packages(),
    author='Praneeth Vasarla',
    description='A tiny neural network library.',
    install_requires=[
        "numpy"
    ],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
