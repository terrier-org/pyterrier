# type: ignore
from typing import List, Union, Literal

import pandas as pd
import pyterrier as pt


class TerrierTextLoader(pt.Transformer):
    """A transformer that loads textual metadata from a Terrier index into a DataFrame by docid or docno."""
    def __init__(
        self,
        index,
        fields: Union[List[str], str, Literal['*']] = '*',
        *,
        verbose=False
    ):
        """Initialise the transformer with the index to load metadata from.

        Args:
            index (pyterrier.terrier.J.Index): The index to load metadata from.
            fields: The fields to load from the index. If '*', all fields will be loaded.
            verbose: Whether to print debug information.
        """
        metaindex = index.getMetaIndex()

        if metaindex is None:
            raise ValueError(f"Index {index} does not have a metaindex")

        available_fields = list(metaindex.getKeys())
        if fields == '*':
            fields = available_fields
        else:
            if isinstance(fields, str):
                fields = [fields]
            missing_fields = set(fields) - set(available_fields)
            if missing_fields:
                raise ValueError(f"Index from {index} did not have requested metaindex keys {list(missing_fields)}. "
                                 f"Keys present in metaindex are {available_fields}")
        self._index = index
        self.metaindex = metaindex
        self.fields = fields
        self.verbose = verbose

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        """Load metadata from the index into the input DataFrame.

        Args:
            inp: The input DataFrame. Must contain either 'docid' or 'docno'.

        Returns:
            A new DataFrame with the metadata columns appended.
        """
        if 'docno' not in inp.columns and 'docid' not in inp.columns:
            raise ValueError(f"Neither docid nor docno are in the input dataframe, found {list(inp.columns)}")

        # Get the docids
        if "docid" not in inp.columns:
            # Look up docids by docno
            docids = inp.docno.map(lambda docno: self.metaindex.getDocument("docno", docno))
        else:
            # Use the provided docids
            docids = inp.docid

        # Look up the metadata and build a new frame to append
        docids = docids.values.tolist() # getItems expects a list
        metadata_matrix = self.metaindex.getItems(self.fields, docids) # indexed by docid then keys
        metadata_frame = pd.DataFrame(metadata_matrix, columns=self.fields)

        # append the input and metadata frames
        inp = inp.drop(columns=self.fields, errors='ignore') # make sure we don't end up with duplicates
        inp = inp.reset_index(drop=True) # reset the index to default (matching metadata_frame)
        return pd.concat([inp, metadata_frame], axis='columns')


@pt.java.required
def terrier_text_loader(
    index,
    fields: Union[List[str], str, Literal['*']] = '*',
    *,
    verbose=False
) -> TerrierTextLoader:
    """Create a transformer that loads textual metadata from a Terrier index into a DataFrame by docid or docno.

    Args:
        index (str or pyterrier.terrier.J.IndexRef or pyterrier.terrier.J.Index): The index to load metadata from.
        fields: The fields to load from the index. If '*', all fields will be loaded.
        verbose: Whether to print debug information.
    """
    if isinstance(index, (str, pt.terrier.J.IndexRef)):
        index = pt.IndexFactory.of(index)
    return TerrierTextLoader(index, fields, verbose=verbose)
