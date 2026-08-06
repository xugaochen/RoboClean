"""Setup script for RoboClean."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="roboclean",
    version="1.0.0",
    description="Universal LeRobot Dataset Cleaning and Visualization Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/roboclean",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pyarrow>=10.0.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "Pillow>=8.3.0",
        "rich>=10.0.0",
    ],
    extras_require={
        "video": ["opencv-python>=4.5.0", "av>=8.0.0"],
        "interactive": ["plotly>=5.3.0"],
    },
    entry_points={"console_scripts": ["roboclean=roboclean.cli.main:main"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
)