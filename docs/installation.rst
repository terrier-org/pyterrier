Installing and Configuring
--------------------------
PyTerrier is a declarative platform for information retrieval experiemnts in Python. It uses the Java-based 
`Terrier information retrieval platform <http://terrier.org>`_ internally to support indexing and retrieval operations.

Pre-requisites
==============

PyTerrier requires Python 3.8 or newer, and Java 11 or newer. PyTerrier is natively supported on Linux, Mac OS X and Windows. 

Installation
============

Installing PyTerrier is easy - it can be installed from the command-line in the normal way using Pip::

    pip install python-terrier


If you want the latest version of PyTerrier, you can install direct from the Github repo::

    pip install --upgrade git+https://github.com/terrier-org/pyterrier.git#egg=python-terrier


NB: There is no need to have a local installation of the Java component, Terrier. PyTerrier will download the latest release on startup.

Troubleshooting
~~~~~~~~~~~~~~~

We aim to ensure that there are pre-compiled binaries available for any dependencies with native components, for all supported Python versions and for all major platforms (Linux, macOS, Windows).
One notable exception is Mac M1 etc., as there are no freely available GitHub Actions runners for M1. Mac M1 installs may require to compile some dependencies.

If the installation failed due to ``pyautocorpus`` did not run successfully, you may need to install ``pcre`` to your machine.

macOS::

    brew install pcre

Linux::

    apt-get update -y
    apt-get install libpcre3-dev -y

For users with an M1 Mac or later models, it is sometimes necessary to install the SSL certificates to avoid certificate errors. 
To do this, locate the `Install Certificates.command` file within the `Application/Python[version]` directory. Once found, double-click on it to run the installation process.

Runing PyTerrier
================

Once installed, you can get going with PyTerrier just by importing it. It's common to alias it as `pt` when importing::

    import pyterrier as pt

Java Configuration
==================

PyTerrier integrates with several Java-based engines, such as Terrier. It uses `PyJnius <https://github.com/kivy/pyjnius>`_ as
a "glue" layer in order to call these Java components. PyTerrier manages this for you automatically and uses reasonable defaults
for the way it interacts with Java. However, sometimes you may need to modify the settings to work with your system. This section
describes how to manage this configuration.

**Note:** Because these options affect the the JVM's settings, they need to be set before Java starts---for instance, at the top of
a script/notebook before any Java components are loaded.

**Starting Java.** PyTerrier will start java when you use a component that requires it, such as ``pt.terrier.Retriever``. However, sometimes
you might want to start it early:

.. autofunction:: pyterrier.java.init

You can also check if Java has been started (either automatically or by `pt.java.init()`):

.. autofunction:: pyterrier.java.started

**Java Home Path.** PyJnius will search the usual places on your machine for a Java installation. If you have problems, you can
overrirde the java home path:

.. autofunction:: pyterrier.java.set_java_home

**Other General Options.** The following are other options for configuring Java:

.. autofunction:: pyterrier.java.add_jar
.. autofunction:: pyterrier.java.add_package
.. autofunction:: pyterrier.java.set_memory_limit
.. autofunction:: pyterrier.java.set_redirect_io
.. autofunction:: pyterrier.java.add_option
.. autofunction:: pyterrier.java.set_log_level

Terrier Configuration
~~~~~~~~~~~~~~~~~~~~~

These options adjust how the Terrier engine is loaded.

.. autofunction:: pyterrier.terrier.set_version
.. autofunction:: pyterrier.terrier.set_helper_version
.. autofunction:: pyterrier.terrier.set_prf_version
.. autofunction:: pyterrier.terrier.set_property
.. autofunction:: pyterrier.terrier.set_properties
.. autofunction:: pyterrier.terrier.extend_classpath

Note on Deprecated Java Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previous versions of PyTerrier required you to run ``pt.init()`` (often in a ``if pt.started()`` block)
to configure Java and start using PyTerrier. This function still exists (it calls ``pt.java.init()``
and associated other configuration methods), but is no longer needed and deprecated.

Instead, PyTerrier now automatically loads Java when a function is called that needs it.
