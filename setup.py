#!/usr/bin/env python
import os

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)


def _load_requirements(path_dir: str = _PATH_ROOT, file_name: str = "requirements.txt") -> list:
    reqs = parse_requirements(open(os.path.join(path_dir, file_name)).readlines())
    return list(map(str, reqs))


setup(
    name="lightning-deepchecks",
    version="0.0.1",
    description="⚡ Deepchecks for Lightning ⚡",
    long_description="⚡ Deepchecks for Lightning ⚡",
    author="Kaushik Bokka",
    author_email="kaushik@lightning.ai",
    url="https://github.com/Lightning-AI/LAI-Deepchecks-Component",
    packages=find_packages(),
    keywords=["deeplearning", "deepchecks", "AI", "validation", "monitoring"],
    python_requires=">=3.8",
    setup_requires=["wheel"],
    install_requires=_load_requirements(),
)
