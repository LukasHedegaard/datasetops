from setuptools import setup, find_packages
setup(
    name="mldatasets",
    version="0.0.2",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        "numpy", "pillow", "pandas", "scipy"
    ],
    extras_require={
        "tests": ["pytest", "pytest-cov", "tensorflow", "pytorch", "flake8"],
        "docs": ["Sphinx", "recommonmark", "sphinx_rtd_theme", "sphinx-autoapi"] 
    }
)
