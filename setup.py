from setuptools import setup, find_packages

__package_name__ = "mphys"
__package_version__ = "1.0.0"


setup(
    name=__package_name__,
    version=__package_version__,
    description=("Components and related code for multiphysics"
        " problems in OpenMDAO"),
    author="",
    author_email="",
    zip_safe=False,
    packages = find_packages(),
    install_requires=[
          'numpy',
          'openmdao>=3.15,!=3.17'
    ],
)
