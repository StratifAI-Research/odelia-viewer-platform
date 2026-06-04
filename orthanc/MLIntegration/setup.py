"""Setup shim for `pip install -e .` — keeps the `shared` module installable.

Dependency pins live in each service's requirements.txt; this file declares no
deps so the root install doesn't pull mutable versions into Docker images.
"""

from setuptools import find_packages, setup

setup(
    name="mlintegration",
    version="2.0.0",
    description="ML Integration package wrapper",
    packages=find_packages(include=["shared", "shared.*"]),
    python_requires=">=3.10",
    install_requires=[],
)
