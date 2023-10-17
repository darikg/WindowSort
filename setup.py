from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='windowsort',
    version='0.1.0',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        'console_scripts': [
            'windowsort=windowsort.gui:main',
        ],
    },
    install_requires=required,
)