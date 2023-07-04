from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

name = "knowledge-neurons"
setup(
    name=name,
    packages=find_packages(),
    version="0.0.2",
    license="MIT",
    description="A library for finding knowledge neurons in pretrained transformer models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/EleutherAI/{name}",
    author="Sid Black",
    author_email="sdtblck@gmail.com",
    install_requires=["transformers", "einops", "numpy", "torch", "seaborn"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
