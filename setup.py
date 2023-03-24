from setuptools import setup, find_packages


# Get version
exec(open('chemistry/version.py', 'r').read())
#
setup(
    name='chemistry',
    version=__version__,
    author='Yisheng Qiu',
    packages=find_packages(),
)
