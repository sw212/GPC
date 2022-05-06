from setuptools import setup, find_packages

setup(
    name="gpc_hmc",
    version="0.0.1",
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages("src", exclude=["*test*"]),
    package_dir={"": "src"},
    install_requires=["jax", "jaxlib"],
    include_package_data=True,
)