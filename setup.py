from setuptools import setup, find_packages
 
setup(
    name = "library", 
    packages = find_packages(),
    extras_require={
    "docs": [
        "sphinx",
        "sphinx_rtd_theme",
        "configargparse",
        ],
    },
    )

