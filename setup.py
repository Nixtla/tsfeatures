import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tsfeatures",
    version="0.0.1",
    author="Federico Garza",
    author_email="fede.garza.ramirez@gmail.com",
    description="features for time series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FedericoGarza/tsfeatures",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'arch',
        'entropy @ git+https://github.com/raphaelvallat/entropy.git',
        'statsmodels',
        'supersmoother',
    ]
)
