from setuptools import setup, find_packages


install_requires = [
    "astropy>=5.2",
    "numpy>=1.23",
    "scipy>=1.10"
    "pandas>=2.0",
    'torch>=2.0',
]

# Get version
exec(open('westlake/version.py', 'r').read())
#
setup(
    name='westlake',
    version=__version__,
    author='Yisheng Qiu',
    author_email="hpc_yqiuu@163.com",
    install_requires=install_requires,
    packages=find_packages(),
    package_data={
        "westlake": ["data/*.pickle"],
    }
)
