#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

requirements = ("numpy",)

setup_requirements = ()

test_requirements = ("pytest",)

description = """A python package for statistical comparison of machine learning models with minimal dependencies. 
                Suports metrics for binary and multiclass classification, and also for regression. 
                You can also implement your own metrics easily using scikit-learn like API."""

setup(
    author="Aleksei Tiulpin",
    author_email="aleksei.tiulpin@oulu.fi",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Researchers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
    ],
    description="A library for statistical model comparison using bootstrap.",
    install_requires=requirements,
    license="MIT license",
    long_description=description,
    include_package_data=True,
    keywords="statistical testing, machine learning",
    name="stambo",
    packages=find_packages(include=["stambo"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/oulu-imeds/stambo",
    version="0.0.1",
    zip_safe=False,
)