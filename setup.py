import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tsfeatures-FedericoGarza", # Replace with your own username
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
    python_requires='>=3.6',
    install_requires=[
        "arch==4.11",
        "git+https://github.com/raphaelvallat/entropy.git@c25553f8b9f8529eb456b93c5dec53c86c779a01#egg=entropy",
        "ESRNN==0.1.2",
        "more-itertools==6.0.0",
        "pandas==0.25.2",
        "rstl==0.1.3",
        "statsmodels==0.11.1",
        "stldecompose==0.0.5",
        "supersmoother==0.4",
        "numpy==1.16.1",
        "git+https://github.com/jakevdp/supsmu"
    ]
)
