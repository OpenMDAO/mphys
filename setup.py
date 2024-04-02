import os
from setuptools import setup, find_packages

__package_name__ = "mphys"
__package_version__ = "1.3.0"

mphys_root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(mphys_root, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

optional_dependencies = {
    'remote': [
        'pbs4py',
        'zmq'
    ],
    'test': [
        'testflo'
    ],
    'docs': [
        'sphinx',
        'sphinxcontrib.bibtex',
        'pbs4py',
        'zmq'
    ]
}
# Add an optional dependency that concatenates all others
optional_dependencies['all'] = sorted([
    dependency
    for dependencies in optional_dependencies.values()
    for dependency in dependencies
])

setup(
    name=__package_name__,
    version=__package_version__,
    description=("Components and related code for multiphysics"
        " problems in OpenMDAO"),
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="",
    author_email="",
    zip_safe=False,
    packages = find_packages(),
    install_requires=[
          'numpy',
          'openmdao >= 3.25, != 3.27.0'
    ],
    extras_require=optional_dependencies,
)
