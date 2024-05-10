from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='windowsort',
    version='0.2.1',
    author='Allen Chen',
    author_email='allenmuhanchen@gmail.com',
    url='https://github.com/EdConnorLab/WindowSort',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        'console_scripts': [
            'windowsort=windowsort.gui:main',
        ],
    },
    install_requires=required,
)