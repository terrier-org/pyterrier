pyterrier package
-----------------

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