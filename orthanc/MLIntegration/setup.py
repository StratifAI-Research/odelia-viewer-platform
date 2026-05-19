"""
Setup script for MLIntegration package
"""
from setuptools import setup, find_packages

setup(
    name="mlintegration",
    version="2.0.0",
    description="ML Integration services for DICOM analysis",
    packages=find_packages(include=['shared', 'shared.*']),
    python_requires='>=3.9',
    install_requires=[
        'flask>=2.0.0',
        'flask-cors>=3.0.0',
        'torch>=1.9.0',
        'torchio>=0.18.0',
        'pydicom>=2.0.0',
        'requests>=2.25.0',
        'numpy>=1.20.0',
        'SimpleITK>=2.0.0',
        'monai>=0.8.0',
        'huggingface-hub>=0.10.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'pytest-mock>=3.10.0',
        ],
    },
)

