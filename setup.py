from setuptools import setup, find_packages

__package_name__ = "omfsi"
__package_version__ = "0.1.0"


setup(
    name=__package_name__,
    version=__package_version__,
    description=("Components and related code for fluid-structure interaction"
        " problems in OpenMDAO"),
    author="",
    author_email="",
    zip_safe=False,
    packages = find_packages()
)