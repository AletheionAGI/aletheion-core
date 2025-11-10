# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Felipe Maya Muniz

"""
Setup configuration for Aletheion Core
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="aletheion-core",
    version="1.0.0",
    author="Felipe M. Muniz",
    author_email="contact@alethea.tech",
    description="Reference implementation of Q₁+Q₂ epistemic gating architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AletheionAGI/aletheion-core",
    project_urls={
        "Paper": "https://doi.org/10.13140/RG.2.2.29925.87527",
        "Documentation": "https://github.com/AletheionAGI/aletheion-core/tree/main/docs",
        "Source": "https://github.com/AletheionAGI/aletheion-core",
        "Tracker": "https://github.com/AletheionAGI/aletheion-core/issues",
    },
    packages=find_packages(),
    classifiers=[
        # Development Status
        "Development Status :: 4 - Beta",
        
        # Intended Audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        
        # Topic
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # License
        "License :: OSI Approved :: GNU Affero General Public License v3",
        
        # Python versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # OS
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=0.990",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    keywords=[
        "machine-learning",
        "artificial-intelligence",
        "epistemic-ai",
        "uncertainty-quantification",
        "internal-coherence",
        "llm",
        "neural-symbolic",
    ],
    license="AGPL-3.0",
    include_package_data=True,
    zip_safe=False,
)
