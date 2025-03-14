from setuptools import setup, find_packages

setup(
    name="singtown_ai",
    version="0.4.2",
    packages=find_packages(),
    install_requires=[
        "pydantic",
        "requests",
    ],
    package_data={
        "singtown_ai": ["result.zip"],
    },
)
