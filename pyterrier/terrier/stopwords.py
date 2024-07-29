from enum import Enum
from typing import Union, List, Dict
import pyterrier as pt


class TerrierStopwords(Enum):
    """
        This enum provides an API for the stopword configuration used during indexing with Terrier
    """

    none = 'none' #: No Stopwords
    terrier = 'terrier' #: Apply Terrier's standard stopword list
    custom = 'custom' #: Apply PyTerrierCustomStopwordList.Indexing for indexing, and PyTerrierCustomStopwordList.Retrieval for retrieval

    @staticmethod
    def _to_obj(this):
        if isinstance(this, list):
            rtr = TerrierStopwords('custom')
            return rtr, list(this)
        try:
            return TerrierStopwords(this), None
        except ValueError:
            return this, None
        
    @staticmethod
    def _indexing_config(this, stopword_list : Union[List[str], None], termpipelines : List[str], properties : Dict[str,str], hooks : List):
        if this is None or this == TerrierStopwords.none:
            pass
        if this == TerrierStopwords.terrier:
            termpipelines.append('Stopwords')
        if this == TerrierStopwords.custom:
            assert pt.terrier.check_version("5.8"), "Terrier 5.8 required"
            assert stopword_list is not None, "expected to receive a stopword list"

            stopword_list_esc = [t.replace(",", "\\,") for t in stopword_list ]

            properties["pyterrier.stopwords"]  = ",".join(stopword_list_esc)
            termpipelines.append('org.terrier.python.PyTerrierCustomStopwordList$Indexing')

            # this hook updates the index's properties to handle the python stopwords list
            def _hook(pyindexer, index):
                pindex = pt.java.cast("org.terrier.structures.PropertiesIndex", index)
                # store the stopwords into the Index's properties
                pindex.setIndexProperty("pyterrier.stopwords", ",".join(stopword_list_esc))

                # change the stopwords list implementation: the Indexing variant obtains
                # stopwords from the global ApplicationSetup properties, while the 
                # Retrieval variant obtains them from the *Index* properties instead
                pindex.setIndexProperty("termpipelines", 
                    pindex.getIndexProperty('termpipelines', None)
                    .replace('org.terrier.python.PyTerrierCustomStopwordList$Indexing',
                             'org.terrier.python.PyTerrierCustomStopwordList$Retrieval'))
                pindex.flush()
            hooks.append(_hook)
