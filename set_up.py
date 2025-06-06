#!/usr/bin/env python3
"""Setup script for the diffusion_lib package."""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'Readme.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "A comprehensive, modular PyTorch-based framework for diffusion models"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        "torch>=1.12.0",
        "torchvision>=0.13.0", 
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "matplotlib>=3.5.0",
        "pillow>=9.0.0"
    ]

setup(
    name="diffusion_lib",
    version="1.0.0",
    description="A comprehensive, modular PyTorch framework for training, sampling, and evaluating diffusion models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Evan Chen",
    author_email="evan.chen@example.com",
    url="https://github.com/evan_chen/diffusion_lib",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    keywords="diffusion models, ddpm, ddim, pytorch, machine learning, generative models, deep learning",
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.900",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-autodoc-typehints>=1.12.0",
            "myst-parser>=0.17.0",
        ],
        "visualization": [
            "wandb>=0.12.0",
            "tensorboard>=2.8.0",
            "seaborn>=0.11.0",
        ],
        "evaluation": [
            "scipy>=1.7.0",
            "scikit-image>=0.18.0",
            "lpips>=0.1.4",
        ],
        "full": [
            "pytest>=6.0", "pytest-cov>=2.0", "black>=21.0", "flake8>=3.8", "mypy>=0.900",
            "sphinx>=4.0", "sphinx-rtd-theme>=1.0", "wandb>=0.12.0", "tensorboard>=2.8.0",
            "scipy>=1.7.0", "scikit-image>=0.18.0", "seaborn>=0.11.0", "lpips>=0.1.4"
        ]
    },
    entry_points={
        "console_scripts": [
            "diffusion-train=diffusion_lib.scripts.train:main",
            "diffusion-sample=diffusion_lib.scripts.sample:main",
            "diffusion-evaluate=diffusion_lib.scripts.evaluate:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/evan_chen/diffusion_lib/issues",
        "Source": "https://github.com/evan_chen/diffusion_lib",
        "Documentation": "https://diffusion-lib.readthedocs.io/",
    },
    include_package_data=True,
    package_data={
        "diffusion_lib": [
            "configs/*.yaml",
            "configs/examples/*.yaml",
        ],
    },
    zip_safe=False,
) 