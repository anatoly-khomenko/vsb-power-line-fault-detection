from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = []

setup(
    name='vsb_power_line_fault_detection',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='VSB Power Line Fault Detection.'
)
