from setuptools import setup, find_packages

setup(
    name="dustgoggles",
    version="0.7.1",
    url="https://github.com/millionconcepts/dustgoggles.git",
    author="Million Concepts",
    author_email="mstclair@millionconcepts.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["cytoolz"],
    extras_require={
        "mosaic": ["pandas", "pyarrow"],
        "pivot": ["pandas"],
        "tests": ["pytest"],
        "array_codecs": ["numpy"]
    }
)
