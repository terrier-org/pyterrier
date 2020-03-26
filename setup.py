import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="python-terrier",
    version="0.1.2",
    author="A-Tsolov",
    author_email="tsolov.aleksandar@gmail.com",
    description="Terrier IR Python API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/A-Tsolov/Pyterrier",
    # packages = setuptools.find_packages(),#['pyterrier'],
    packages=[""],
    package_dir={"":"pyterrier"},
    # packages=setuptools.find_packages(),
    # packages=setuptools.find_packages(include=['pyterrier']),
    # packages=["pyterrier"],
    # py_modules=["pyterrier.pyterrier","pyterrier.batchretrieve", "pyterrier.utils", "pyterrier.index"],
    # py_modules=["pyterrier.pyterrier","pyterrier.batchretrieve", "pyterrier.utils", "pyterrier.index"],
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
    ],
    python_requires='>=3.6',
)
