"""Validation utilities for checking the inputs of transformers."""
import warnings
from types import TracebackType
from typing import List, Optional, Type, Union, Set
import pandas as pd
import pyterrier as pt


class _TransformerMode:
    def __init__(self, missing_columns: List[str], extra_columns: List[str], mode_name: Optional[str] = None):
        self.missing_columns = missing_columns
        self.extra_columns = extra_columns
        self.mode_name = mode_name

    def __str__(self):
        return f'{self.mode_name} (missing: {self.missing_columns}, extra: {self.extra_columns})'

    def __repr__(self):
        return f'TransformerMode(missing_columns={self.missing_columns!r}, ' \
               f'extra_columns={self.extra_columns!r}, ' \
               f'mode_name={self.mode_name!r})'


class InputValidationError(KeyError):
    """Exception raised when input validation fails."""
    def __init__(self, message: str, modes: List[_TransformerMode]):
        """Create an InputValidationError."""
        assert len(modes) > 0
        super().__init__(message)
        self.modes = modes

    def __str__(self):
        return self.args[0] + ' ' + str(self.modes)

    def __repr__(self):
        return f'InputValidationError({self.args[0]!r}, {self.modes!r})'


class InputValidationWarning(Warning):
    """Warning raised when input validation fails in warn mode."""
    pass


def columns(
    inp: Union[pd.DataFrame, List[str]],
    *,
    includes: Optional[List[str]] = None,
    excludes: Optional[List[str]] = None,
    warn: bool = False
) -> None:
    """Check that the input frame has the expected columns.

    Args:
        inp: Input DataFrame or columns to validate
        includes: List of required columns
        excludes: List of forbidden columns
        warn: If True, raise warnings instead of exceptions for validation errors

    Raises:
        InputValidationError: If warn=False and validation fails
        InputValidationWarning: If warn=True and validation fails

    .. versionchanged:: 0.15.0
        Accept ``List[str]`` inp columns
    """
    with any(inp, warn=warn) as v:
        v.columns(includes=includes, excludes=excludes)


def query_frame(inp: Union[pd.DataFrame, List[str]], extra_columns: Optional[List[str]] = None, warn: bool = False) -> None:
    """Check that the input frame is a valid query frame.

    Args:
        inp: Input DataFrame or columns to validate
        extra_columns: Additional required columns
        warn: If True, raise warnings instead of exceptions for validation errors

    Raises:
        InputValidationError: If warn=False and validation fails
        InputValidationWarning: If warn=True and validation fails

    .. versionchanged:: 0.15.0
        Accept ``List[str]`` inp columns
    """
    with any(inp, warn=warn) as v:
        v.query_frame(extra_columns)


def result_frame(inp: Union[pd.DataFrame, List[str]], extra_columns: Optional[List[str]] = None, warn: bool = False) -> None:
    """Check that the input frame is a valid result frame.

    Args:
        inp: Input DataFrame or columns to validate
        extra_columns: Additional required columns
        warn: If True, raise warnings instead of exceptions for validation errors

    Raises:
        InputValidationError: If warn=False and validation fails
        InputValidationWarning: If warn=True and validation fails

    .. versionchanged:: 0.15.0
        Accept ``List[str]`` inp columns
    """
    with any(inp, warn=warn) as v:
        v.result_frame(extra_columns)


def document_frame(inp: Union[pd.DataFrame, List[str]], extra_columns: Optional[List[str]] = None, warn: bool = False) -> None:
    """Check that the input frame is a valid document frame.

    Args:
        inp: Input DataFrame or columns to validate
        extra_columns: Additional required columns
        warn: If True, raise warnings instead of exceptions for validation errors

    Raises:
        InputValidationError: If warn=False and validation fails
        InputValidationWarning: If warn=True and validation fails

    .. versionchanged:: 0.15.0
        Accept ``List[str]`` inp columns
    """
    with any(inp, warn=warn) as v:
        v.document_frame(extra_columns)


def columns_iter(
    inp: 'pt.utils.PeekableIter',
    *,
    includes: Optional[List[str]] = None,
    excludes: Optional[List[str]] = None,
    warn: bool = False
) -> None:
    """Check that the input frame has the expected columns.

    Args:
        inp: Input DataFrame to validate
        includes: List of required columns
        excludes: List of forbidden columns
        warn: If True, raise warnings instead of exceptions for validation errors

    Raises:
        InputValidationError: If warn=False and validation fails
        InputValidationWarning: If warn=True and validation fails
    """
    with any_iter(inp, warn=warn) as v:
        v.columns(includes=includes, excludes=excludes)


def query_iter(inp: 'pt.utils.PeekableIter', extra_columns: Optional[List[str]] = None, warn: bool = False) -> None:
    """Check that the input iterator is a valid query iterator.

    Args:
        inp: Input iterator to validate
        extra_columns: Additional required columns
        warn: If True, raise warnings instead of exceptions for validation errors
    Raises:
        InputValidationError: If warn=False and validation fails
        InputValidationWarning: If warn=True and validation fails
    """
    with any_iter(inp, warn=warn) as v:
        v.query_iter(extra_columns)


def result_iter(inp: 'pt.utils.PeekableIter', extra_columns: Optional[List[str]] = None, warn: bool = False) -> None:
    """Check that the input iterator is a valid result iterator.

    Args:
        inp: Input iterator to validate
        extra_columns: Additional required columns
        warn: If True, raise warnings instead of exceptions for validation errors
    Raises:
        InputValidationError: If warn=False and validation fails
        InputValidationWarning: If warn=True and validation fails
    """
    with any_iter(inp, warn=warn) as v:
        v.result_iter(extra_columns)


def document_iter(inp: 'pt.utils.PeekableIter', extra_columns: Optional[List[str]] = None, warn: bool = False) -> None:
    """Check that the input iterator is a valid document iterator.

    Args:
        inp: Input iterator to validate
        extra_columns: Additional required columns
        warn: If True, raise warnings instead of exceptions for validation errors
    Raises:
        InputValidationError: If warn=False and validation fails
        InputValidationWarning: If warn=True and validation fails
    """
    with any_iter(inp, warn=warn) as v:
        v.document_iter(extra_columns)


def any(inp: Union[pd.DataFrame, List[str]], warn: bool = False) -> '_ValidationContextManager':
    """Create a validation context manager for a DataFrame or list of columns to test multiple possible modes.

    Args:
        inp: Input DataFrame or list of columns to validate
        warn: If True, raise warnings instead of exceptions for validation errors
    """
    return _ValidationContextManager(inp, warn=warn)


def any_iter(inp: 'pt.utils.PeekableIter', warn: bool = False) -> '_IterValidationContextManager':
    """Create a validation context manager for an iterator to test multiple possible modes.

    Args:
        inp: Input iterator to validate
        warn: If True, raise warnings instead of exceptions for validation errors
    """
    if not isinstance(inp, pt.utils.PeekableIter):
        raise AttributeError('inp is not peekable. Run the following before calling this function.\n'
                             'inp = pt.utils.peekable(inp) # !! IMPORTANT: you must re-assign the input to peekable '
                             '(not just pass it in), otherwise you will skip the first record !!')
    return _IterValidationContextManager(inp, warn=warn)


class _ValidationContextManager:
    """Context manager for validating the input to transformers."""
    def __init__(self, inp: Union[pd.DataFrame, List[str]], warn: bool = False):
        """Create a ValidationContextManager for the given DataFrame."""
        if isinstance(inp, pd.DataFrame):
            self.inp_columns = list(inp.columns)
        else:
            self.inp_columns = inp
        self.mode: Optional[str] = None
        self.attempts = 0
        self.errors: List[_TransformerMode] = []
        self.warn = warn

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType]
    ) -> Optional[bool]:

        if exc_type is not None:
            return False # the captured exception takes priority

        if self.attempts > 0 and self.attempts == len(self.errors):
            message = "DataFrame does not match required columns for this transformer."
            if self.warn:
                warnings.warn(f"{message} {self.errors}", InputValidationWarning)
            else:
                raise InputValidationError(message, self.errors)
        return None

    def columns(self,
                *,
                includes: Optional[List[str]] = None,
                excludes: Optional[List[str]] = None,
                mode: Optional[str] = None) -> bool:
        """Check that the input frame has the ``includes`` columns and doesn't have the ``excludes`` columns."""
        includes = includes if includes is not None else []
        excludes = excludes if excludes is not None else []
        missing_columns = set(includes) - set(self.inp_columns)
        extra_columns = set(excludes) & set(self.inp_columns)
        self.attempts += 1

        if missing_columns or extra_columns:
            self.errors.append(_TransformerMode(
                missing_columns=[c for c in includes if c in missing_columns],
                extra_columns=[c for c in excludes if c in extra_columns],
                mode_name=mode,
            ))
            return False

        if self.mode is None and mode is not None:
            self.mode = mode

        return True

    def query_frame(self, extra_columns: Optional[List[str]] = None, mode: Optional[str] = None) -> bool:
        """Check that the input frame is a valid query frame, with optional extra columns."""
        extra_columns = list(extra_columns) if extra_columns is not None else []
        return self.columns(includes=['qid'] + extra_columns, excludes=['docno'], mode=mode)

    def result_frame(self, extra_columns: Optional[List[str]] = None, mode: Optional[str] = None) -> bool:
        """Check that the input frame is a valid result frame, with optional extra columns."""
        extra_columns = list(extra_columns) if extra_columns is not None else []
        return self.columns(includes=['qid', 'docno'] + extra_columns, mode=mode)

    def document_frame(self, extra_columns: Optional[List[str]] = None, mode: Optional[str] = None) -> bool:
        """Check that the input frame is a valid document frame, with optional extra columns."""
        extra_columns = list(extra_columns) if extra_columns is not None else []
        return self.columns(includes=['docno'] + extra_columns, excludes=['qid'], mode=mode)


_EMPTY_ITER = object()

class _IterValidationContextManager:
    def __init__(self, inp: 'pt.utils.PeekableIter', warn: bool = False):
        self.sample_cols: Union[Set[str], object]
        try:
            self.sample_cols = set(inp.peek().keys())
        except StopIteration:
            self.sample_cols = _EMPTY_ITER
        self.mode: Optional[str] = None
        self.attempts = 0
        self.errors: List[_TransformerMode] = []
        self.warn = warn

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType]
    ) -> Optional[bool]:

        if exc_type is not None:
            return False # the captured exception takes priority

        if self.attempts > 0 and self.attempts == len(self.errors):
            message = "Input does not match required columns for this transformer."
            if self.warn:
                warnings.warn(f"{message} {self.errors}", InputValidationWarning)
            else:
                raise InputValidationError(message, self.errors)
        return None

    def columns(self,
                *,
                includes: Optional[List[str]] = None,
                excludes: Optional[List[str]] = None,
                mode: Optional[str] = None) -> bool:
        self.attempts += 1
        includes = includes if includes is not None else []
        excludes = excludes if excludes is not None else []
        if not isinstance(self.sample_cols, set):
            assert self.sample_cols == _EMPTY_ITER # AKA, but need to check type to make mypy happy
            self.errors.append(_TransformerMode(
                missing_columns=list(includes),
                extra_columns=[],
                mode_name=mode,
            ))
            return False
        missing_columns = set(includes) - self.sample_cols
        extra_columns = set(excludes) & self.sample_cols

        if missing_columns or extra_columns:
            self.errors.append(_TransformerMode(
                missing_columns=[c for c in includes if c in missing_columns],
                extra_columns=[c for c in excludes if c in extra_columns],
                mode_name=mode,
            ))
            return False

        if self.mode is None and mode is not None:
            self.mode = mode

        return True

    def query_iter(self, extra_columns: Optional[List[str]] = None, mode: Optional[str] = None) -> bool:
        """Check that the input iterator is a valid query iterator, with optional extra columns."""
        extra_columns = list(extra_columns) if extra_columns is not None else []
        return self.columns(includes=['qid'] + extra_columns, excludes=['docno'], mode=mode)

    def result_iter(self, extra_columns: Optional[List[str]] = None, mode: Optional[str] = None) -> bool:
        """Check that the input iterator is a valid result iterator, with optional extra columns."""
        extra_columns = list(extra_columns) if extra_columns is not None else []
        return self.columns(includes=['qid', 'docno'] + extra_columns, mode=mode)

    def document_iter(self, extra_columns: Optional[List[str]] = None, mode: Optional[str] = None) -> bool:
        """Check that the input iterator is a valid document iterator, with optional extra columns."""
        extra_columns = list(extra_columns) if extra_columns is not None else []
        return self.columns(includes=['docno'] + extra_columns, excludes=['qid'], mode=mode)

    def empty(self, *, mode: Optional[str] = 'empty'):
        self.attempts += 1
        if self.sample_cols != _EMPTY_ITER:
            self.errors.append(_TransformerMode(
                missing_columns=[],
                extra_columns=[],
                mode_name=mode,
            ))
            return False

        if self.mode is None and mode is not None:
            self.mode = mode
        return True
