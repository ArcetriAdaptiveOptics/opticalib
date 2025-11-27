import os
from setuptools import setup, find_packages
from setuptools.command.install import install

about = {}
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "opticalib", "__version__.py"), "r") as _:
    exec(_.read(), about)

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name=about["__title__"],
    version=about["__version__"],
    author=about["__author__"],
    maintainer=about["__maintainer__"],
    author_email=about["__author_email__"],
    description=about["__description__"],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url=about["__url__"],
    license=about["__license__"],
    packages=find_packages(),
    # Ensure standalone module used by console_scripts is packaged into wheels
    py_modules=["setup_calpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "opticalib": [
            "core/_configurations/configuration.yaml",
            "simulator/_API/AdOpticaData/*",
            "simulator/_API/alpao_conf.yaml",
        ]
    },
    entry_points={
        "console_scripts": [
            "calpy=setup_calpy:main",
        ],
    },
)
