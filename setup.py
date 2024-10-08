import re

from setuptools import find_packages, setup


def get_version():
    with open("src/__init__.py", "r") as f:
        for line in f:
            match = re.match(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", line)
            if match:
                return match.group(1)
    raise RuntimeError("Version not found in __init__.py")


setup(version=get_version(), packages=find_packages())
