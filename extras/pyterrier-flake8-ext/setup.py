import os
from setuptools import setup

def get_version(rel_path):
    suffix = os.environ["PYTERRIER_VERSION_SUFFIX" ] if "PYTERRIER_VERSION_SUFFIX" in os.environ else ""
    for line in open(rel_path):
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1] + suffix
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="pyterrier-flake8-ext",
    version=get_version("pyterrier_flake8_ext/__init__.py"),
    author="Sean MacAvaney",
    author_email='sean.macavaney@glasgow.ac.uk',
    description="Terrier IR platform Python API",
    project_urls={
        'Documentation': 'https://pyterrier.readthedocs.io',
        'Changelog': 'https://github.com/terrier-org/pyterrier/releases',
        'Issue Tracker': 'https://github.com/terrier-org/pyterrier/issues',
        'CI': 'https://github.com/terrier-org/pyterrier/actions',
    },
    long_description=open('README.md', 'rt').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/terrier-org/pyterrier",
    packages=['pyterrier_flake8_ext'],
    entry_points={
        'flake8.extension': [
            'PT100 = pyterrier_flake8_ext:JavaCheck',
        ]
    },
    install_requires=[
        'flake8',
    ],
    python_requires='>=3.8',
)
