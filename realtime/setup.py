#!/usr/bin/env python3
"""
Setup script for Technical Architecture Python package
"""

from setuptools import find_packages, setup

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="technical_architecture",
    version="0.1.0",
    author="Sheel Morjaria",
    author_email="sheelmorjaria@gmail.com",
    description="Zero-copy Rust-Python integration for high-performance data processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sheelmorjaria/birdsong_analysis",
    packages=find_packages(),
    package_data={
        "technical_architecture": ["*.py"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Creative Commons Attribution No "
        "Derivatives 4.0 International (CC BY-ND 4.0)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Rust",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    zip_safe=False,
)
