import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="plexus",
    version="0.0.1",
    author="Arlo J. Proctor",
    author_email="nexus.lib.project@gmail.com",
    description="A biologically-inspired Python library for neurochemical and deep learning systems supporting AGI development.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nexus-lib-project/plexus-python",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU LESSER GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
)
