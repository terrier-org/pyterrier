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

            pip install pyterrier[all]

        NB: If you dont require the full Terrier retriever functionality, you can install a smaller version of PyTerrier without all
        the dependencies by running:   

        .. code-block:: bash

            pip install pyterrier

        NBB: You can still use the older pypi name of ``python-terrier`` if you wish.

    .. tab:: From GitHub

        If you want the latest version of PyTerrier, you can install directly from the Github repository

        .. code-block:: bash

            pip install --upgrade git+https://github.com/terrier-org/pyterrier.git


**Problems Installing?** Check out the :doc:`installation troubleshooting guide <troubleshooting/installation>`.

Running PyTerrier
=================

Once installed, you can get going with PyTerrier just by importing it. It's common to alias it as ``pt`` when importing::

    import pyterrier as pt


Java is required for some functionality in PyTerrier. If you want to check to make sure Java is installed and configured properly, you
can run::

    pt.java.init()

Note that it is not longer required to run ``pt.init()`` before using PyTerrier.

**Problems with Java?** Check out the :doc:`Java troubleshooting guide <troubleshooting/java>`.
