Making a PyTerrier Extension Package
=====================================================

.. note::

   This section of the documentation is under development.

Cookiecutter
-----------------------------------------------------

You can build a starter Python extension package using `cookiecutter <https://github.com/cookiecutter/cookiecutter>`_.
Run the following command and follow the instructions in the terminal:

.. code-block::

   pip install -U cookiecutter
   cookiecutter https://github.com/seanmacavaney/cookiecutter-pyterrier.git

This builds a package based on the template in `seanmacavaney/cookiecutter-pyterrier <https://github.com/seanmacavaney/cookiecutter-pyterrier>`_.

Documentation
-----------------------------------------------------

You can add your extension package here in PyTerrier's docs. This can help people find your features.

**Step 1 (optional): Embed documentation in your package**

If you want to embed documentation pages directly in 

**Step 2: Make a pull request to extensions.txt**

You can make a pull request to the `extensions.txt <https://github.com/terrier-org/pyterrier/blob/master/docs/extensions.txt>`_ file.

If your package includes embedded documentation (from Step 1), add a line like:

.. code-block::

   package_id # PackageName <PackageURL>

If your package does not embed documentation, add a line like:

.. code-block::

   # PackageName <PackageURL>


Where ``package_id`` is the package ID on pypi, ``PackageName`` is the human-readable name, and ``PackageURL`` is the URL of the repository
of the extension (or another suitable external documentation page).
