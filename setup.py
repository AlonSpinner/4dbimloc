from setuptools import setup
from setuptools import find_packages

setup(
    name = "bim4loc",
    version = "1.0.0",
    description = "bla bla",
    author = "bla bla",
    url = "https://github.com/AlonSpinner/4dbimloc",
    packages = find_packages(exclude = ('tests*')),
    )
