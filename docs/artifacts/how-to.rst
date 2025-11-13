Artifact How-To Guides
--------------------------------------------------

This page provides answers to common questions about using PyTerrier artifacts,
including how to share and load them using various services.


---------------------------------------------------------------------

.. _artifacts:how-to:huggingface:

How do I load or share an artifact on HuggingFace?
=====================================================================

.. related:: pyterrier.Artifact.to_hf
.. related:: pyterrier.Artifact.from_hf

The `HuggingFace Hub <https://huggingface.co/docs/hub>`__ is a popular platform for sharing models, datasets, and other research artifacts.  
You can upload and download PyTerrier artifacts directly through the :class:`~pyterrier.Artifact` API.

You can load artifacts from HuggingFace Hub using :meth:`~pyterrier.Artifact.from_hf`, and share your own artifacts using :meth:`~pyterrier.Artifact.to_hf`,
as shown in the examples below.

**Load an artifact from HuggingFace**

.. code-block:: python
   :caption: Load from HuggingFace Hub

   import pyterrier as pt

   index = pt.Artifact.from_hf('username/myindex')

This will download and return a :class:`~pyterrier.Artifact` object of the correct type.

**Share an artifact to HuggingFace**

.. code-block:: python
   :caption: Upload to HuggingFace Hub

   import pyterrier as pt

   artifact = ...  # e.g., pt.Artifact.load('path/to/index')
   artifact.to_hf('username/myindex')

This command builds and uploads a HuggingFace-compatible artifact package.  
A template ``README.md`` is created automatically—complete it to describe your artifact.

.. note::
   You must have the ``huggingface-hub`` package installed and set the environment variable ``HF_TOKEN`` with "write" permission.  
   See the `HuggingFace authentication guide <https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication>`__ for details.

**Extras**

* Find all PyTerrier artifacts on HuggingFace using the `pyterrier-artifact <https://huggingface.co/datasets?other=pyterrier-artifact>`__ tag.
* You can also load directly from URL syntax: ``pt.Artifact.from_url('hf:username/repo')``.
* Large artifacts are automatically split to comply with HuggingFace file-size limits.
* The default branch is ``main``. Use ``branch=`` or ``username/repo@branch`` to specify another branch.


---------------------------------------------------------------------

.. _artifacts:how-to:zenodo:

How do I load or share an artifact on Zenodo?
=====================================================================

.. related:: pyterrier.Artifact.to_zenodo
.. related:: pyterrier.Artifact.from_zenodo

`Zenodo <https://zenodo.org/>`__ is a long-term repository for research outputs.  
You can use it to preserve and share your PyTerrier artifacts with persistent DOIs.

You can load artifacts from Zenodo using :meth:`~pyterrier.Artifact.from_zenodo`, and share your own artifacts using :meth:`~pyterrier.Artifact.to_zenodo`,
as shown in the examples below.

**Load an artifact from Zenodo**

.. code-block:: python
   :caption: Load from Zenodo

   import pyterrier as pt

   index = pt.Artifact.from_zenodo('artifact-id')  # e.g., "13839452"

This fetches the artifact package from Zenodo and returns a corresponding :class:`~pyterrier.Artifact`.

**Share an artifact to Zenodo**

.. code-block:: python
   :caption: Upload to Zenodo

   import pyterrier as pt

   artifact = ...  # e.g., pt.Artifact.load('path/to/index')
   artifact.to_zenodo()

After running this, a draft upload is created.  
:meth:`~pyterrier.Artifact.to_zenodo` prints the URL — complete the form (e.g., add title, authors) and publish it to make it public.

.. note::
   Set your ``ZENODO_TOKEN`` environment variable with "deposit:write" and "deposit:actions" permissions.  
   See `Zenodo authentication docs <https://developers.zenodo.org/#authentication>`__ for details.

**Extras**

* Browse all PyTerrier artifacts on Zenodo via the `pyterrier-artifact <https://zenodo.org/search?q=metadata.subjects.subject%3A%22pyterrier-artifact%22>`__ tag.
* Load from a URL using: ``pt.Artifact.from_url('zenodo:artifact-id')``.


---------------------------------------------------------------------

.. _artifacts:how-to:p2p:

How do I directly share an artifact with a collaborator (Peer-to-Peer)?
==========================================================================

.. related:: pyterrier.Artifact.to_p2p
.. related:: pyterrier.Artifact.from_p2p

If you need to share a work-in-progress artifact privately, PyTerrier supports peer-to-peer (P2P) transfers
through :meth:`~pyterrier.Artifact.to_p2p` and :meth:`~pyterrier.Artifact.from_p2p`.  These methods use
`magic-wormhole <https://magic-wormhole.readthedocs.io/>`__ for secure one-time exchanges.

**Start sharing from the host machine**

.. code-block:: python
   :caption: Host a P2P artifact

   import pyterrier as pt

   artifact = ...  # e.g., pt.Artifact.load('path/to/my_index')
   artifact.to_p2p()

This builds the artifact package and prints a one-time sharing code, such as ``xx-xxx-xxx``.

**Receive the artifact on the target machine**

.. code-block:: python
   :caption: Receive a P2P artifact

   import pyterrier as pt

   artifact = pt.Artifact.from_p2p('xx-xxx-xxx', 'my_index')

The command will securely transfer and load the artifact.

.. note::
   You must install the ``magic-wormhole`` package to use P2P sharing.


---------------------------------------------------------------------

.. _artifacts:how-to:share-other:

How do I share artifacts using other services?
=====================================================================

.. related:: pyterrier.Artifact.build_package
.. related:: pyterrier.Artifact.from_url

You can host an artifact package anywhere—such as Dropbox, Google Drive, or institutional storage.

You will first need to build the artifact distribution package using :meth:`~pyterrier.Artifact.build_package`, then upload the resulting file
to your desired service. Artifacts shared this way can be loaded using :meth:`~pyterrier.Artifact.from_url`.

**Build a distributable artifact package**

.. code-block:: python
   :caption: Build an artifact package

   import pyterrier as pt

   artifact = ...  # e.g., pt.Artifact.load('path/to/index')
   package_path = artifact.build_package('artifact.tar.lz4')

This creates an archive file you can upload manually.

**Load from a hosted package**

.. code-block:: python
   :caption: Load from a URL

   import pyterrier as pt

   artifact = pt.Artifact.from_url('https://example.com/artifact.tar.lz4')

As long as the URL points to a valid artifact distribution package, it can be loaded seamlessly.


--------------------------------------------------------------------

.. _artifacts:how-to:services:

What services can I use to share PyTerrier artifacts?
=====================================================================

PyTerrier supports several ways to share and load artifacts — including :ref:`HuggingFace <artifacts:how-to:huggingface>`,
:ref:`Zenodo <artifacts:how-to:zenodo>`, and :ref:`peer-to-peer transfers <artifacts:how-to:p2p>`.
transfers. You can also host artifacts on :ref:`other services <artifacts:how-to:share-other>` by manually building
and uploading the artifact package.


---------------------------------------------------------------------

What types of artifacts are supported?
=====================================================================

Please refer to the :doc:`listing` for a full listing of available Artifact classes in PyTerrier.


---------------------------------------------------------------------

How do I implement new Artifact classes?
=====================================================================

Please refer to :doc:`../extending/artifacts` for guidance on implementing new Artifact types.


--------------------------------------------------------------------

What is an Artifact?
=====================================================================

"Artifact" often refers to a broad range of items. For
instance, the `ACM defines  <https://www.acm.org/publications/policies/artifact-review-and-badging-current>`__
an artifact as: "a digital object that was either created by the authors to be used as part of the study
or generated by the experiment itself."

In PyTerrier, we use a narrower definition. We treat artifacts as components that
can be represented as a file or directory stored on disk. These are most frequently built indexes,
but can also be resources such as cached pipeline results.
