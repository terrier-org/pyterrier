from enum import Enum
import pyterrier as pt


_stemmer_cache = {}


class TerrierStemmer(Enum):
    """
        This enum provides an API for the stemmers available in Terrier. The stemming configuration is saved in the index
        and loaded at retrieval time. `Snowball <https://snowballstem.org/>`_ stemmers for various languages 
        `are available in Terrier <http://terrier.org/docs/current/javadoc/org/terrier/terms/package-summary.html>`_.

        It can also be used to access the stemmer::

            stemmer = pt.TerrierStemmer.porter
            stemmed_word = stemmer.stem('abandoned')

    """
    none = 'none' #: Apply no stemming
    porter = 'porter' #: Apply Porter's English stemmer
    weakporter = 'weakporter' #: Apply a weak version of Porter's English stemmer
    # available snowball stemmers in Terrier
    danish = 'danish' #: Snowball Danish stemmer
    finnish = 'finnish' #: Snowball Finnish stemmer
    german = 'german' #: Snowball German stemmer
    hungarian = 'hungarian' #: Snowball Hungarian stemmer
    norwegian = 'norwegian' #: Snowball Norwegian stemmer
    portugese = 'portugese' #: Snowball Portuguese stemmer
    swedish = 'swedish' #: Snowball Swedish stemmer
    turkish = 'turkish' #: Snowball Turkish stemmer

    @staticmethod
    def _to_obj(this):
        try:
            return TerrierStemmer(this)
        except ValueError:
            return this

    @staticmethod
    def _to_class(this):
        if this is None or this == TerrierStemmer.none:
            return None
        if this == TerrierStemmer.porter:
            return 'PorterStemmer'
        if this == TerrierStemmer.weakporter:
            return 'WeakPorterStemmer'
        
        # snowball stemmers
        if this == TerrierStemmer.danish:
            return 'DanishSnowballStemmer'
        if this == TerrierStemmer.finnish:
            return 'FinnishSnowballStemmer'
        if this == TerrierStemmer.german:
            return 'GermanSnowballStemmer'
        if this == TerrierStemmer.hungarian:
            return 'HungarianSnowballStemmer'
        if this == TerrierStemmer.norwegian:
            return 'NorwegianSnowballStemmer'
        if this == TerrierStemmer.portugese:
            return 'PortugueseSnowballStemmer'
        if this == TerrierStemmer.swedish:
            return 'SwedishSnowballStemmer'
        if this == TerrierStemmer.turkish:
            return 'TurkishSnowballStemmer'

        if isinstance(this, str):
            return this

    @pt.java.required
    def stem(self, tok):
        if self not in _stemmer_cache:
            clz_name = self._to_class(self)
            if clz_name is None:
                _stemmer_cache[self] = _NoOpStem()
            else:
                if '.' not in clz_name:
                    clz_name = f'org.terrier.terms.{clz_name}'
                 # stemmers are termpipeline objects, and these have chained constructors
                 # pass None to use the appropriate constructor
                _stemmer_cache[self] = pt.java.autoclass(clz_name)(None)
        return _stemmer_cache[self].stem(tok)


class _NoOpStem():
    def stem(self, word):
        return word
