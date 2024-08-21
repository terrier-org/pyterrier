import pandas as pd
import pyterrier as pt


class TerrierTextLoader(pt.Transformer):
    def __init__(self, index, fields = '*', *, verbose=False):
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

    def transform(self, inp):
        if 'docno' not in inp.columns and 'docid' not in inp.columns:
            raise ValueError(f"Neither docid nor docno are in the input dataframe, found {list(inp.columns)}")

        # Get the docids
        if "docid" not in inp.columns:
            # Look up docids by docno
            docids = TerrierDocidLoader.docnos_to_docids(self.metaindex, inp.docno)
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


class TerrierDocidLoader(pt.Transformer):
    def __init__(self, index, fields, *, verbose=False):
        metaindex = index.getMetaIndex()
        if metaindex is None:
            raise ValueError(f"Index {index} does not have a metaindex")
        self._index = index
        self.metaindex = metaindex

    def transform(self, inp):
        if 'docno' not in inp.columns:
            raise ValueError(f"docno are in the input dataframe, found {list(inp.columns)}")
        return inp.assign(docid=self.docnos_to_docids(self.metaindex, inp.docno))

    @staticmethod
    def docnos_to_docids(metaindex, docnos: pd.Series) -> pd.Series:
        return docnos.map(lambda docno: metaindex.getDocument("docno", docno))


@pt.java.required
def terrier_text_loader(index, fields = '*', *, verbose=False):
    if isinstance(index, (str, pt.terrier.J.IndexRef)):
        index = pt.IndexFactory.of(index)
    return TerrierTextLoader(index, fields, verbose=verbose)
