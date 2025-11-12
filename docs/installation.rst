Installation
--------------------------
PyTerrier is a declarative platform for building information retrieval pipelines and conducting
experiemnts in Python.

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

            pip install 'pyterrier[all]' # :footnote: :nocomment: ``python-terrier`` is a shortcut for ``pyterrier[all]``

        If you want a minimal installation without optional dependencies, you can install just the core package:

        .. code-block:: bash

            pip install pyterrier

    .. tab:: From GitHub

        If you want the latest version of PyTerrier, you can install directly from the Github repository

        .. code-block:: bash

            pip install --upgrade git+https://github.com/terrier-org/pyterrier.git


**Problems Installing?** Check out the :doc:`installation troubleshooting guide <troubleshooting/installation>`.

Running PyTerrier
=================

Once installed, you can get going with PyTerrier just by importing it. It's common to alias it as ``pt``:

.. code-block:: python

    import pyterrier as pt

.. admonition:: Optional: Check Java Installation
    :class: note, dropdown

    Java is required for some functionality in PyTerrier. If you want to check to make sure Java is installed and configured properly, you
    can run:

    .. code-block:: python

        pt.java.init()

    Note that it was previously required to run ``init()`` before using PyTerrier. This is no longer required.

    **Problems with Java?** Check out the :doc:`Java troubleshooting guide <troubleshooting/java>`.
