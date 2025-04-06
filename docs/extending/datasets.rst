Extending PyTerrier with New Datasets
=====================================================

.. note::

    This guide is for adding new datasets to PyTerrier, allowing them to be easily used by others.
    If you simply want to run PyTerrier with your own data, you can build Pandas DataFrames compatible
    with the :doc:`PyTerrier Data Model <../datamodel>`.

    If you want to use existing built-in datasets, you can find them on :doc:`this page <../datasets>`.

You can add new datasets to PyTerrier through the Datasets API. This involves:

1. Creating a new ``Dataset`` class, which contains the logic for processing your data into the :doc:`PyTerrier Data Model <../datamodel>`.
2. Creating a new ``DatasetProvider`` class, which provides access to your datasets.
3. Registering the new dataset provider with PyTerrier using an entry point.

Each of these steps is detailed below.


``Dataset`` Classes
-----------------------------------------------------

Your ``Dataset`` class should inherit from :class:`pyterrier.datasets.Dataset`. Most commonly, the class will implement
the following methods:

- ``get_topics()``: Returns the topics for the dataset.
- ``get_qrels()``: Returns the qrels for the dataset.
- ``get_corpus_iter()``: Returns an iterator over the corpus for the dataset.

For example:

.. code-block:: python
    :caption: Example ``Dataset`` Implementation

    from typing import Iterable, Dict, Any
    import pyterrier as pt
    import pandas as pd

    class MyDataset(pt.datasets.Dataset):
        def get_topics(self, variant=None) -> pd.DataFrame:
            # Logic to load and return topics
            return pd.DataFrame(read_topics('my_topics'))

        def get_qrels(self, variant=None) -> pd.DataFrame:
            # Logic to load and return qrels
            return pd.DataFrame(read_qrels('my_qrels'))

        def get_corpus_iter(self, verbose=True) -> Iterable[Dict[str, Any]]:
            # Logic to load and return corpus iterator
            for line in open('my_file'):
                docno, text = parse_line(line)
                yield {'docno': docno, 'text': text}


``DatasetProvider`` Classes
-----------------------------------------------------

Your ``DatasetProvider`` class provides access to the datasets you want to include in your package.
It should inherit from ``pt.datasets.DatasetProvider`` and implement the following methods:

- ``get_dataset(name)``: Returns a specific dataset by name.
- ``list_dataset_names()``: Returns a list of the names (IDs) of all datasets provided by this object.

For example:

.. code-block:: python
    :caption: Example ``DatasetProvider`` Implementation

    import pyterrier as pt

    class MyDatasetProvider(pt.datasets.DatasetProvider):
        def get_dataset(self, name):
            if name == "my_dataset":
                return MyDataset()
            else:
                raise ValueError(f"Dataset {name} not found")

        def list_dataset_names(self):
            return ["my_dataset"]


Registering your ``DatasetProvider``
-----------------------------------------------------

You can register your ``DatasetProvider`` with PyTerrier using an entry point in your package's ``setup.py`` file or
``pyproject.toml`` file. This allows PyTerrier to discover your datasets when your package is installed.

The entry point should provide a prefix that identifies your dataset provider. When a user requests a dataset with a
name that starts with this prefix, PyTerrier will use your ``DatasetProvider`` to load the dataset. For example, if you
register your provider with the prefix ``my_prefix``, if a user requests the dataset ``pt.get_dataset("my_prefix:my_dataset")``,
PyTerrier will load your ``MyDatasetProvider`` class and invoke its ``get_dataset("my_dataset")`` method.

If you are using a ``setup.py`` file, you can add the following entry point as follows:

.. code-block:: python
    :caption: Example Dataset Provider Entry Point in ``setup.py``

    from setuptools import setup

    setup(
        ... # <-- the rest of your configuration
        entry_points={
            "pyterrier.dataset_provider": [ # <-- PyTerrier looks for this entry point
                "my_prefix = my_package.MyDatasetProvider" # <-- when a dataset looks like 'my_prefix:{name}', it will load MyDatasetProvider
            ]
        },
    )

If you are using ``pyproject.toml``, you can add the entry point as follows:

.. code-block:: toml
    :caption: Example Dataset Provider Entry Point in ``pyproject.toml``

    ... # <-- the rest of your configuration

    [project.entry-points."pyterrier.dataset_provider"] # <-- PyTerrier looks for this entry point
    "my_prefix" = "my_package.MyDatasetProvider" # <-- when a dataset looks like 'my_prefix:{name}', it will load MyDatasetProvider
