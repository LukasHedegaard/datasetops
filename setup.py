from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="datasetops",
    version="0.0.5",
    author="Lukas Hedegaard",
    description="Fluent dataset operations, compatible with your favorite libraries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LukasHedegaard/datasetops",
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Framework :: Pytest",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["numpy", "pillow", "pandas", "scipy"],
    extras_require={
        "tests": [
            "pytest",
            "pytest-cov",
            "flake8",
            "tensorflow",
            "torch",
            "torchvision",
        ],
        "docs": ["Sphinx", "recommonmark", "sphinx_rtd_theme", "sphinx-autoapi"],
        "build": ["setuptools", "wheel", "twine"],
    },
)
