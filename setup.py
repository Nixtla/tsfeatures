import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tsfeatures",
    version="0.3.1",
    description="Calculates various features from time series data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FedericoGarza/tsfeatures",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "antropy>=0.1.4",
        "arch>=4.11",
        "pandas>=1.0.5",
        "scikit-learn>=0.23.1",
        "statsmodels>=0.12.2",
        "supersmoother>=0.4"
    ]
)
