import os
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

    def find_packages(where='.'):
        # os.walk -> list[(dirname, list[subdirs], list[files])]
        return [folder.replace("/", ".").lstrip(".")
                for (folder, _, fils) in os.walk(where)
                if "__init__.py" in fils]
with open("README.md", "r") as fh:
    long_description = fh.read()

# see https://packaging.python.org/guides/single-sourcing-package-version/
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


requirements = []
with open('requirements.txt', 'rt') as f:
    for req in f.read().splitlines():
        if req.startswith("#"):
            continue
        if req.startswith('git+'):
            pkg_name = req.split('/')[-1].replace('.git', '')
            if "#egg=" in pkg_name:
                pkg_name = pkg_name.split("#egg=")[1]
            requirements.append(f'{pkg_name} @ {req}')
        else:
            requirements.append(req)
    if not "nt" in os.name:
        requirements.append("pytrec_eval")

setup(
    name="python-terrier",
    version=get_version("pyterrier/__init__.py"),
    author="Craig Macdonald",
    author_email='craigm@dcs.gla.ac.uk',
    description="Terrier IR platform Python API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/terrier-org/pyterrier",
    packages=['pyterrier'] + ['pyterrier.' + i for i in find_packages('pyterrier')],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    python_requires='>=3.6',
)
