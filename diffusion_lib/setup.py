from setuptools import setup, find_packages

setup(
    name="diffusion_lib",
    version="0.1.0",
    packages=find_packages(),
    description="A Python package for diffusion model training",
    author="Your Name",
    author_email="your.email@example.com",
    install_requires=["numpy", "matplotlib", "torch", "scikit-learn", "tqdm"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)