Troubleshooting Java
--------------------------------------------

PyTerrier integrates with several Java-backed engines, including Terrier. It uses `PyJnius <https://github.com/kivy/pyjnius>`_ as
a "glue" layer in order to call these Java components. PyTerrier manages this automatically for you (including downloading required
JAR files, running the required Java "initialization" step, etc.), and uses reasonable defaults for the way it interacts with Java.

Typically you only need :ref:`Java installed <install_java>`, but in some cases you may also need :ref:`to adjust the Java configuration <java_configuration>`.

Installing Java
==================

.. _install_java:

On **MacOS with Homebrew**, you can run ``brew install openjdk``

On **Ubuntu/Debian Linux**, you can run ``sudo apt install openjdk-11-jdk-headless``

For other systems, or of the above commands do not work, we recommend the `Java Developer Resources Guide <https://www.java.com/en/download/help/download_options.html>`__.

Java Configuration
==================

.. _java_configuration:

This section describes how to manage Java's configuration in PyTerrier.

.. note::

    Because these options affect the the JVM's settings, they need to be set before Java starts---for instance, at the top of a
    script/notebook before any Java components are loaded.

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
.. autofunction:: pyterrier.terrier.set_property
.. autofunction:: pyterrier.terrier.set_properties
.. autofunction:: pyterrier.terrier.extend_classpath

Note on Deprecated Java Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previous versions of PyTerrier required you to run ``pt.init()`` (often in a ``if pt.started()`` block)
to configure Java and start using PyTerrier. This function still exists (it calls ``pt.java.init()``
and associated other configuration methods), but is no longer needed and deprecated.

Instead, PyTerrier now automatically loads Java when a function is called that needs it.
