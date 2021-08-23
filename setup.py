from setuptools import setup, find_packages

setup(
    name="dustgoggles",
    version="0.1.2",
    url="https://github.com/millionconcepts/dustgoggles.git",
    author="Million Concepts",
    author_email="mstclair@millionconcepts.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["cytoolz"],
    extras_require={"pivot": ["numpy", "pandas"]}
)
