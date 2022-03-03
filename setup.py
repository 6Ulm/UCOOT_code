import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ucoot_code",
    version="0.0.1",
    author="Huy Tran, Alexis Thual",
    description="Python implementation of Unbalanced Fused Gromow Wasserstein",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/6Ulm/UCOOT_code",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["torch", "sklearn"],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
