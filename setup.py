from setuptools import find_packages, setup
from lendingclubdb4ml import __version__

setup(
    name="lendingclubdb4ml",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    version=__version__,
    description="",
    author=""
)
