import pandas as pd
from typing import Union, List, Literal
from typing import Iterable
import pyterrier as pt


class IRDSDatasetProvider(pt.datasets.DatasetProvider):
    def get_dataset(self, name) -> pt.datasets.Dataset:
        return IRDSDataset(name)

    def list_dataset_names(self) -> Iterable[str]:
        import ir_datasets
        return list(ir_datasets.registry)


class IRDSDataset(pt.datasets.Dataset):
    def __init__(self, irds_id):
        import ir_datasets
        self._irds_id = irds_id
        self._irds_ref = ir_datasets.load(self._irds_id)

    def irds_ref(self):
        return self._irds_ref

    def get_corpus_iter(self, verbose=True, start=0, count=None):
        ds = self.irds_ref()
        if not ds.has_docs():
            raise NotImplementedError(f"{self!r} doesn't support get_corpus_iter")
        it = ds.docs_iter()
        total = ds.docs_count()

        # use slicing if requested
        if start > 0 or count is not None:
            if count is not None:
                it = it[start:start+count]
                total = count
            else:
                it = it[start:]
                total -= start

        # tqdm support
        if verbose:
            it = pt.tqdm(it, desc=f'{self._irds_id} documents', total=total)

        # rewrite to follow pyterrier std
        def gen():
            for doc in it:
                doc = doc._asdict()
                # pyterrier uses "docno"
                doc['docno'] = doc.pop('doc_id')
                yield doc

        # provide a __len__ (e.g., so tqdm shows progress)
        return pt.utils.GeneratorLen(gen(), total)

    def get_corpus_lang(self):
        ds = self.irds_ref()
        if not ds.has_docs():
            return None
        return ds.docs_lang()

    def get_topics(self, variant=None):
        """
            Returns the topics, as a dataframe, ready for retrieval. 
        """
        ds = self.irds_ref()
        if not ds.has_queries():
            raise NotImplementedError(f"{self._irds_id} doesn't support get_topics")
        qcls = ds.queries_cls()
        assert variant is None or variant in qcls._fields[1:], f"{self._irds_id} only supports the following topic variants {qcls._fields[1:]}"
        df = pd.DataFrame(ds.queries_iter())

        df.rename(columns={"query_id": "qid"}, inplace=True) # pyterrier uses "qid"

        if variant is not None:
            # Some datasets have a query field called "query". We need to remove it or
            # we'll end up with multiple "query" columns, which will cause problems
            # because many components are written assuming no columns have the same name.
            if variant != 'query' and 'query' in df.columns:
                df.drop(columns=['query'], axis=1, inplace=True)
            df.rename(columns={variant: "query"}, inplace=True) # user specified which version of the query they want
            df.drop(columns=df.columns.difference(['qid','query']), axis=1, inplace=True)
        elif len(qcls._fields) == 2:
            # auto-rename single query field to "query" if there's only query_id and that field
            df.rename(columns={qcls._fields[1]: "query"}, inplace=True)
        else:
            print(f'There are multiple query fields available: {qcls._fields[1:]}. To use with pyterrier, provide variant or modify dataframe to add query column.')
        return df

    def get_topics_lang(self):
        ds = self.irds_ref()
        if not ds.has_queries():
            return None
        return ds.queries_lang()

    def get_qrels(self, variant=None):
        """ 
            Returns the qrels, as a dataframe, ready for evaluation.
        """
        ds = self.irds_ref()
        if not ds.has_qrels():
            raise NotImplementedError(f"{self._irds_id} doesn't support get_qrels")
        qrelcls = ds.qrels_cls()
        qrel_fields = [f for f in qrelcls._fields if f not in ('query_id', 'doc_id', 'iteration')]
        assert variant is None or variant in qrel_fields, f"{self._irds_id} only supports the following qrel variants {qrel_fields}"
        df = pd.DataFrame(ds.qrels_iter())

        # pyterrier uses "qid" and "docno"
        df.rename(columns={
            "query_id": "qid",
            "doc_id": "docno"}, inplace=True)

        # pyterrier uses "label"
        if variant is not None:
            df.rename(columns={variant: "label"}, inplace=True)
        if len(qrel_fields) == 1:
            # usually "relevance"
            df.rename(columns={qrel_fields[0]: "label"}, inplace=True)
        elif 'relevance' in qrel_fields:
            print(f'There are multiple qrel fields available: {qrel_fields}. Defaulting to "relevance", but to use a different one, supply variant')
            df.rename(columns={'relevance': "label"}, inplace=True)
        else:
            print(f'There are multiple qrel fields available: {qrel_fields}. To use with pyterrier, provide variant or modify dataframe to add query column.')

        return df

    def get_results(self, variant=None) -> pd.DataFrame:
        """ 
            Returns a standard result set provided by the dataset. This is useful for re-ranking experiments.
        """
        ds = self.irds_ref()
        if not ds.has_scoreddocs():
            raise NotImplementedError(f"{self._irds_id} doesn't support get_results")
        result = pd.DataFrame(ds.scoreddocs)
        result = result.rename(columns={'query_id': 'qid', 'doc_id': 'docno'}) # convert irds field names to pyterrier names
        result.sort_values(by=['qid', 'score', 'docno'], ascending=[True, False, True], inplace=True) # ensure data is sorted by qid, -score, did
        # result doesn't yet contain queries (only qids) so load and merge them in
        topics = self.get_topics(variant)
        result = pd.merge(result, topics, how='left', on='qid')
        return result

    def _describe_component(self, component):
        ds = self.irds_ref()
        if component == "topics":
            if ds.has_queries():
                fields = ds.queries_cls()._fields[1:]
                if len(fields) > 1:
                    return list(fields)
                return True
            return None
        if component == "qrels":
            if ds.has_qrels():
                fields = [f for f in ds.qrels_cls()._fields if f not in ('query_id', 'doc_id', 'iteration')]
                if len(fields) > 1:
                    return list(fields)
                return True
            return None
        if component == "corpus":
            return ds.has_docs() or None
        if component == "results":
            return ds.has_scoreddocs() or None
        return None

    def info_url(self):
        top_id = self._irds_id.split('/', 1)[0]
        suffix = f'#{self._irds_id}' if top_id != self._irds_id else ''
        return f'https://ir-datasets.com/{top_id}.html{suffix}'

    def __repr__(self):
        return f"IRDSDataset({self._irds_id!r})"

    def text_loader(
        self,
        fields: Union[List[str], str, Literal['*']] = '*',
        *,
        verbose: bool = False,
    ) -> pt.Transformer:
        """Create a transformer that loads text fields from an ir_datasets dataset into a DataFrame by docno.

        Args:
            fields: The fields to load from the dataset. If '*', all fields will be loaded.
            verbose: Whether to print debug information.
        """
        return IRDSTextLoader(self, fields, verbose=verbose)


class IRDSTextLoader(pt.Transformer):
    """A transformer that loads text fields from an ir_datasets dataset into a DataFrame by docno."""
    def __init__(
        self,
        dataset: IRDSDataset,
        fields: Union[List[str], str, Literal['*']] = '*',
        *,
        verbose=False
    ):
        """Initialise the transformer with the index to load metadata from.

        Args:
            dataset: The dataset to load text from.
            fields: The fields to load from the dataset. If '*', all fields will be loaded.
            verbose: Whether to print debug information.
        """
        if not dataset.irds_ref().has_docs():
            raise ValueError(f"Dataset {dataset} does not provide docs")
        docs_cls = dataset.irds_ref().docs_cls()

        available_fields = [f for f in docs_cls._fields if f != 'doc_id' and docs_cls.__annotations__[f] is str]
        if fields == '*':
            fields = available_fields
        else:
            if isinstance(fields, str):
                fields = [fields]
            missing_fields = set(fields) - set(available_fields)
            if missing_fields:
                raise ValueError(f"Dataset {dataset} did not have requested metaindex keys {list(missing_fields)}. "
                                 f"Keys present in metaindex are {available_fields}")

        self.dataset = dataset
        self.fields = fields
        self.verbose = verbose

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        """Load text fields from the dataset into the input DataFrame.

        Args:
            inp: The input DataFrame. Must contain 'docno'.

        Returns:
            A new DataFrame with the text columns appended.
        """
        if 'docno' not in inp.columns:
            raise ValueError(f"input missing 'docno' column, available columns: {list(inp.columns)}")
        irds = self.dataset.irds_ref()
        docstore = irds.docs_store()
        docnos = inp.docno.values.tolist()

        # Load the new data
        fields = ['doc_id'] + self.fields
        set_docnos = set(docnos)
        it = (tuple(getattr(doc, f) for f in fields) for doc in docstore.get_many_iter(set_docnos))
        if self.verbose:
            it = pt.tqdm(it, unit='d', total=len(set_docnos), desc='IRDSTextLoader') # type: ignore
        metadata = pd.DataFrame(list(it), columns=fields).set_index('doc_id')
        metadata_frame = metadata.loc[docnos].reset_index(drop=True)

        # append the input and metadata frames
        inp = inp.drop(columns=self.fields, errors='ignore') # make sure we don't end up with duplicates
        inp = inp.reset_index(drop=True) # reset the index to default (matching metadata_frame)
        return pd.concat([inp, metadata_frame], axis='columns')
