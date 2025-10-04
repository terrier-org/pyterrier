from pyterrier.datasets._core import Dataset, DatasetProvider, get_dataset, find_datasets, list_datasets, transformer_from_dataset, DATASET_MAP, RemoteDataset

# _builtin and _irds is not loaded here -- they are loaded on-demand once a dataset is requested that needs it.

__all__ = ["Dataset", "DatasetProvider", "get_dataset", "find_datasets", "list_datasets", "transformer_from_dataset", "DATASET_MAP", "RemoteDataset"]
