import setuptools

# parse version number from _version.py without importing flare
import re
VERSIONFILE="flare/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# get description and dependencies
with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    dependencies = fh.readlines()

setuptools.setup(
    name="mir-flare",
    packages=setuptools.find_packages(exclude=["tests"]),
    version=verstr,
    author="Materials Intelligence Research",
    author_email="mir@g.harvard.edu",
    description="Fast Learning of Atomistic Rare Events",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mir-group/flare",
    python_requires=">=3.6",
    install_requires=dependencies,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 4 - Beta",
    ],
)
