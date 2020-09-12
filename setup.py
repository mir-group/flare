import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    dependencies = fh.readlines()

setuptools.setup(
    name="mir-flare",
    packages=setuptools.find_packages(exclude=["tests"]),
    version="0.1.1",
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
