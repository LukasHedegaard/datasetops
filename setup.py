from setuptools import setup, find_packages
setup(
    name="mldatasets",
    version="0.0.1",
    packages=find_packages('src'),
    package_dir={'': 'src'},

    install_requires=[
        "pytest-cov", "numpy", "pandas", "pillow", "scipy", "pytest", "Sphinx", "recommonmark", "sphinx_rtd_theme", "sphinx-autoapi"
    ]
)
