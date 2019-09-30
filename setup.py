import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("LICENSE", "r") as fh:
    license = fh.read()

with open("requirements.txt", "r") as fh:
    dependencies = fh.readlines()

setuptools.setup(
    name="flare",
    packages=setuptools.find_packages(exclude=["tests"]),
    version="0.0.1",
    author="Materials Intelligence Research",
    author_email="mir@g.harvard.edu",
    description="Fast Learning of Atomistic Rare Events",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mir-group/flare",
    python_requires=">=3.6",
    install_requires=dependencies,
    license=license,
)
