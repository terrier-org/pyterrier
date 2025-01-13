"""Module for inspecting pyterrier objects."""
from typing import Tuple, Type, Union

import pyterrier as pt


def artifact_type_format(artifact: Union[Type, 'pt.Artifact']) -> Tuple[str, str]:
    """Returns the type and format of the specified artifact.

    These values are sourced by either the ARTIFACT_TYPE and ARTIFACT_FORMAT constants of the artifact, or (if these
    are not available) by matching on the entry points.

    Returns:
        A tuple containing the artifact's type and format.

    Raises:
        TypeError: If the artifact's type or format could not be determined.
    """
    artifact_type, artifact_format = None, None

    # Source #1: ARTIFACT_TYPE and ARTIFACT_FORMAT constants
    if hasattr(artifact, 'ARTIFACT_TYPE') and hasattr(artifact, 'ARTIFACT_FORMAT'):
        artifact_type = artifact.ARTIFACT_TYPE
        artifact_format = artifact.ARTIFACT_FORMAT

    # Source #2: entry point name
    if artifact_type is None or artifact_format is None:
        for entry_point in pt.utils.entry_points('pyterrier.artifact'):
            if artifact.__module__.split('.')[0] != entry_point.value.split(':')[0].split('.')[0]:
                continue # only try loading entry points that share the same top-level module
            entry_point_cls = entry_point.load()
            if isinstance(artifact, type) and artifact == entry_point_cls or isinstance(artifact, entry_point_cls):
                artifact_type, artifact_format = entry_point.name.split('.', 1)
                break

    if artifact_type is None or artifact_format is None:
        raise TypeError(f'{artifact} does not provide type and format (either as constants or via entry point)')

    return artifact_type, artifact_format
