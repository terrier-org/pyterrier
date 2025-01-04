# type: ignore
from typing import Union, List
import pyterrier as pt


@pt.java.required
class IndexFactory:
    """
    The ``of()`` method of this factory class allows to load a Terrier `Index <http://terrier.org/docs/current/javadoc/org/terrier/structures/Index.html>`_.

    NB: This class "shades" the native Terrier `IndexFactory <http://terrier.org/docs/current/javadoc/org/terrier/structures/IndexFactory.html>`_ class - it offers essential the same API,
    except that the ``of()`` method contains a memory kwarg, that can be used to load additional index data structures into memory. 

    Terrier data structures that can be loaded into memory:
     - 'inverted' - the inverted index, contains posting lists for each term. In the default configuration, this is read in from disk in chunks.
     - 'lexicon' - the dictionary. By default, a binary search of the on-disk structure is used, so loading into memory can enhance speed.
     - 'meta' - metadata about documents. Used as the final stage of retrieval, one seek for each retrieved document.
     - 'direct' - contains posting lists for each document. No speed advantage for loading into memory unless pseudo-relevance feedback is being used.
     - 'document' - contains document lengths, which are anyway loaded into memory. No speed advantage for loading into memory unless pseudo-relevance feedback is being used.
    """

    @staticmethod
    def _load_into_memory(index, structures=['lexicon', 'direct', 'inverted', 'meta'], load=False):

        REWRITES = {
            'meta' : {
                # both metaindex implementations have the same property
                'org.terrier.structures.ZstdCompressedMetaIndex' : {
                    'index.meta.index-source' : 'fileinmem',
                    'index.meta.data-source' : 'fileinmem'},
            
                'org.terrier.structures.CompressingMetaIndex' : {
                    'index.meta.index-source' : 'fileinmem',
                    'index.meta.data-source' : 'fileinmem'}
            },
            'lexicon' : {
                'org.terrier.structures.FSOMapFileLexicon' : {
                    'index.lexicon.data-source' : 'fileinmem'
                }
            },
            'direct' : {
                'org.terrier.structures.bit.BitPostingIndex' : {
                    'index.direct.data-source' : 'fileinmem'}
            },
            'inverted' : {
                'org.terrier.structures.bit.BitPostingIndex' : {
                    'index.inverted.data-source' : 'fileinmem'}
            },
        }
        if "direct" in structures:
            REWRITES['document'] = {
                # we have to be sensitive to the presence of fields or not
                # NB: loading these structures into memory only benefit direct index access
                'org.terrier.structures.FSADocumentIndex' : {
                    'index.document.class' : 'FSADocumentIndexInMem'
                }, 
                'org.terrier.structures.FSAFieldDocumentIndex' : {
                    'index.document.class' : 'FSADocumentIndexInMemFields'
                }
            }

        pindex = pt.java.cast("org.terrier.structures.IndexOnDisk", index)
        load_profile = pindex.getIndexLoadingProfileAsRetrieval()
        dirty_structures = set()
        for s in structures:
            if not pindex.hasIndexStructure(s):
                continue
            clz = pindex.getIndexProperty(f"index.{s}.class", "notfound")
            if clz not in REWRITES[s]:
                raise ValueError(f"Cannot load structure {s} into memory, underlying class {clz} is not supported")

            # we only reload an index structure if a property has changed
            dirty = False
            for k, v in REWRITES[s][clz].items():
                if pindex.getIndexProperty(k, "notset") != v:
                    pindex.setIndexProperty(k, v)
                    dirty_structures.add(s)

                    # if the document index is reloaded, the inverted index should be reloaded too
                    # NB: the direct index needs reloaded too, but this option is only available IF
                    # the direct index is setup
                    if s == "document":
                        dirty_structures.add("inverted")

        # remove the old data structures from memory
        for s in dirty_structures:
            if pindex.structureCache.containsKey(s):
                pindex.structureCache.remove(s)

        # force the index structures to be loaded now
        if load:
            for s in dirty_structures:
                pindex.getIndexStructure(s)

        # dont allow the index properties to be rewritten
        pindex.dirtyProperties = False
        return index

    @staticmethod 
    def of(indexlike, memory : Union[bool, List[str]] = False):
        """
        Loads an index. Returns a Terrier `Index <http://terrier.org/docs/current/javadoc/org/terrier/structures/Index.html>`_ object.

        Args:
            indexlike(str or IndexRef): Where is the index located
            memory(bool or List[str]): If the index should be loaded into memory. Use `True` for all structures, or a list of structure names.
        """
        load_profile = pt.terrier.J.IndexOnDisk.getIndexLoadingProfileAsRetrieval()

        if memory or (isinstance(memory, list) and len(memory) > 0): #MEMORY CAN BE A LIST?
            pt.terrier.J.IndexOnDisk.setIndexLoadingProfileAsRetrieval(False)
        index = pt.terrier.J.IndexFactory.of(indexlike)
        
        # noop if memory is False
        pt.terrier.J.IndexOnDisk.setIndexLoadingProfileAsRetrieval(load_profile)
        if not memory:
            return index
        if isinstance(memory, list):
            return IndexFactory._load_into_memory(index, structures=memory)
        return IndexFactory._load_into_memory(index)
