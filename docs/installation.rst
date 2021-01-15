Installing and Configuring
--------------------------
PyTerrier is a declarative platform for information retrieval experiemnts in Python. It uses the Java-based 
`Terrier information retrieval platform <http://terrier.org>`_ internally to support indexing and retrieval operations.

Pre-requisites
==============

PyTerrier requires Python 3.6 or newer, and Java 11 or newer. 

PyTerrier is natively supported on Linux and Mac OS X. PyTerrier uses `Pytrec_eval <https://github.com/cvangysel/pytrec_eval>`_ for evaluation, 
and the latter does not install automatically on Windows.

Installation
============

Installing PyTerrier is easy - it can be installed from the command-line in the normal way using Pip:

    pip install python-terrier


If you want the latest version of PyTerrier, you can install direct from the Github repo::

    pip install --upgrade git+https://github.com/terrier-org/pyterrier.git#egg=python-terrier


NB: There is no need to have a local installation of the Java component, Terrier. PyTerrier will download the latest release on startup.

Configuration
==============

You must always start by importing PyTerrier and running init()::

    import pyterrier as pt
    pt.init()

PyTerrier uses `PyJnius <https://github.com/kivy/pyjnius>`_ as a "glue" layer in order to call Terrier's Java classes. PyJnius will search 
the usual places on your machine for a Java installation. If you have problems, set the `JAVA_HOME` environment variable::

    import os
    os.environ["JAVA_HOME"] = "/path/to/my/jdk"
    import pyterrier as pt
    pt.init()

`pt.init()` has a multitude of options, for instance that can make PyTerrier more notebook friendly, or to change the underlying version of Terrier, as described below.

API Reference
=============

All usages of PyTerrier start by importing PyTerrier and starting it using the `init()` method::

    import pyterrier as pt
    pt.init()

PyTerrier uses some of the functionality of the Java-based `Terrier <http://terrier.org>`_ IR platform
for indexing and retrieval functionality. Calling `pt.init()` downloads, if necessary, the Terrier jar 
file, and starts the Java Virtual Machine (JVM). It also configures the Terrier so that it can be more 
easily used from Python, such as redirecting the stdout and stderr streams, logging level etc.

Below, there is more documentation about method related to starting Terrier using PyTerrier,
and ways to change the configuration.

Startup-related methods
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pyterrier.init()

.. autofunction:: pyterrier.started()

.. autofunction:: pyterrier.run()

Methods to change PyTerrier configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pyterrier.extend_classpath()

.. autofunction:: pyterrier.logging()

.. autofunction:: pyterrier.redirect_stdouterr()

.. autofunction:: pyterrier.set_property()

.. autofunction:: pyterrier.set_properties()

.. autofunction:: pyterrier.set_tqdm()