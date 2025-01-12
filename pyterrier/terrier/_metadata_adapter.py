from typing import List


def terrier_artifact_metadata_adapter(path: str, dir_listing: List[str]):
    """Guess whether this path is a terrier index.

    Some terrier indexes are missing pt_meta.json, but we can assume they are terrier indexes based on the
    presence of a data.properties file.
    """
    if 'data.properties' in dir_listing:
        return {
            'type': 'sparse_index',
            'format': 'terrier',
            'package_hint': 'python-terrier',
        }
