from setuptools import find_packages, setup

setup(
    name="autoinnovator",
    author="TIG labs",
    version="0.1",
    description="Automated LLM innovator",
    license="MIT",
    install_requires=[
        'numpy', 
        'scipy', 
    ],
    packages=find_packages(),
)