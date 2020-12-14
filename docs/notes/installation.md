# Installing and Configuring PyTerrier

## Pre-requisites

PyTerrier requires Python 3.6 or newer, and Java 11 or newer. 

PyTerrier is natively supported on Linux and Mac OS X. PyTerrier uses [Pytrec_eval](https://github.com/cvangysel/pytrec_eval) extensively, and the latter does not install automatically on Windows.

## Installation

Installing PyTerrier is easy - it can be installed from the command-line in the normal way:
```
pip install python-terrier
```

If you want the latest version of PyTerrier, you can install direct from the Github repo:
```
pip install --upgrade git+https://github.com/terrier-org/pyterrier.git#egg=python-terrier
```

NB: There is no need to have a local installation of the Java component, Terrier. PyTerrier will download the latest release on startup.

## Configuration

You must always start by importing PyTerrier and running init():
```python
import pyterrier as pt
pt.init()
```

PyTerrier uses [PyJnius](https://github.com/kivy/pyjnius) as a "glue" layer in order to call Terrier's Java classes. PyJnius will search the usual places on your machine for a Java installation. If you have problems, set the `JAVA_HOME` environment variable:
```python
import os
os.environ["JAVA_HOME"] = "/path/to/my/jdk"
import pyterrier as pt
pt.init()
```

`pt.init()` has a multitude of options, for instance that can make PyTerrier more notebook friendly, or to change the underlying version of Terrier. See the API reference for `pt.init()` for more information.