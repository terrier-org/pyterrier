import pyterrier as pt

class TerrierIndex(pt.Artifact):
    def __init__(self, path):
        self.path = path
        self._cache = {}

    def retriever(self, *, controls=None, properties=None, metadata=["docno"],  num_results=None, wmodel=None, threads=1):
        return pt.TerrierRetrieve(self, controls, properties, metadata, num_results, wmodel)

    def bm25(self, *, num_results=1000, threads=1):
        return self.retriever(wmodel='BM25', num_results=num_results, threads=threads)

    @classmethod
    def _try_load(cls, path, metadata):
        if metadata['format'] == 'terrier':
            return cls(path)

    def __repr__(self):
        return f'TerrierIndex({pt.artifact.path_repr(self.path)})'

    def get_index_ref(self):
        if 'index_ref' not in self._cache:
            self._cache['index_ref'] = pt.IndexRef.of(self.path)
        return self._cache['index_ref']
