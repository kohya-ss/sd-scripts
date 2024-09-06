from setuptools import setup, find_packages

setup(
    name="sd-scripts",
    packages=find_packages(),  # Automatically find all packages in the repository
    package_dir={"sd-scripts": "."},  # Install all packages under sd-scripts in the venv
)
