from setuptools import setup, find_packages


# Get version
exec(open('westlake/version.py', 'r').read())
#
setup(
    name='westlake',
    version=__version__,
    author='Yisheng Qiu',
    packages=find_packages(),
)
