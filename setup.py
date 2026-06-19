from setuptools import setup

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

setup(
    name="scientific_data_analysis",
    version="2.0.6",
    author="Michael Weide",
    author_email="michael.weide08@gmail.com",
    description="Library of some functions I find useful for data analysis.",
    long_description=long_description,
    url="https://github.com/Atix95/Data_analysis_template_python",
    license="unlicense",
    install_requires=[
        "numpy",
        "matplotlib",
        "pytexit",
        "scipy",
        "sympy",
        "uncertainties",
    ],
    packages=["scientific_data_analysis"],
)
