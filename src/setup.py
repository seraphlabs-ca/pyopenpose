#!/usr/bin/env shebang-python

from setuptools import find_packages
from common.setuptools_support import setup

setup(
    name="attention_seeker",
    packages=find_packages(include=['attention_seeker']),
    scripts=[],

    install_requires=[],

    # allows to disable all cpp parsing
    disable_cpp=True,
    # if True SWIG modules are copied, else soft link is created
    swig_build_copy=False,

    # install SWIG modules {module_name: (source_path, target_path)
    swig_modules={
        'attention_seekerSWIG': ('attention_seeker', 'attention_seeker'),
    },

    # # metadata for upload to PyPI
    description="Human detection and pose estimation package. TVision Insight Tools",
    long_description="",
    keywords="",
    url="",
    author="Micha Livne",
    author_email="livne@seraphlabs.ca",
    license="TVision Insight proprietary code.",
)
