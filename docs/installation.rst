Installing and Configuring
--------------------------
PyTerrier is a declarative platform for information retrieval experiemnts in Python. It uses the Java-based 
`Terrier information retrieval platform <http://terrier.org>`_ internally to support indexing and retrieval operations.

Pre-requisites
==============

PyTerrier requires Python 3.9 or newer. Java 11 or newer is required for some functionality.
PyTerrier is natively supported on Linux, Mac OS X and Windows. 


Installation
============

Installing PyTerrier is easy - it can be installed from the command-line using ``pip``:

.. tabs::

    .. tab:: From PyPi

        .. code-block:: bash

            pip install python-terrier

    .. tab:: From GitHub

        If you want the latest version of PyTerrier, you can install directly from the Github repository

        .. code-block:: bash

            pip install --upgrade git+https://github.com/terrier-org/pyterrier.git#egg=python-terrier


.. admonition:: Problems Installing?

   Check out the :doc:`installation troubleshooting guide <troubleshooting/installation>`.

Runing PyTerrier
================

Once installed, you can get going with PyTerrier just by importing it. It's common to alias it as ``pt`` when importing::

    import pyterrier as pt


Java is required for some functionality in PyTerrier. If you want to check to make sure Java is installed and configured properly, you
can run::

    pt.java.init()

.. admonition:: Problems with Java?

   Check out the :doc:`Java troubleshooting guide <troubleshooting/java>`.
