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

setup(
    name="python-terrier",
    version="0.1.3",
    author="Craig Macdonald",
    author_email='craigm{at}.dcs.gla.ac.uk',
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
    # data_files=[('my_data', ['../terrier-project-5.1-jar-with-dependencies.jar'])],
    install_requires=[
        "pyjnius",
        "numpy",
        "pandas",
        "wget",
        "pytrec_eval",
        "tqdm",
        "deprecation"
    ],
    python_requires='>=3.6',
)
