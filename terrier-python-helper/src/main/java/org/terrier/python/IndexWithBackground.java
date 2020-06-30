package org.terrier.python;

import org.terrier.querying.IndexRef;
import org.terrier.structures.*;
import java.util.Iterator;
import java.util.Map;
import java.io.IOException;

public class IndexWithBackground extends Index {

    Index parent;
    Index background;
    ProxyLexicon proxLex;
    
    public IndexWithBackground(Index _parent, Index _background) {
        this.parent = _parent;
        this.background = _background;
        this.proxLex = new ProxyLexicon(parent.getLexicon(), background.getLexicon());
    }

    @Override 
    public PostingIndex<?> getDirectIndex() {
        return this.parent.getDirectIndex();
    }

    @Override
    public DocumentIndex getDocumentIndex() {
        return this.parent.getDocumentIndex();
    }

    @Override public String toString() {
        return "background:" + parent.getIndexRef().toString() + "," + background.getIndexRef().toString();
    }

    @Override public IndexRef getIndexRef() {
        return Index.makeDirectIndexRef(this);
    }

    @Override public PostingIndex<?> getInvertedIndex() {
        return this.parent.getInvertedIndex();
    }

    @Override
    public CollectionStatistics getCollectionStatistics()
    {
        return this.background.getCollectionStatistics();
    }

    @Override
    public MetaIndex getMetaIndex() {
        return this.parent.getMetaIndex();
    }

    @Override
    public Lexicon<String> getLexicon() {
        return this.proxLex;
    }


    static class ProxyLexicon extends Lexicon<String>
    {
        Lexicon<String> pLexicon;
        Lexicon<String> bLexicon;

        public ProxyLexicon(Lexicon<String> p, Lexicon<String> b) {
            this.pLexicon = p;
            this.bLexicon = b;
        }

        public int numberOfEntries() {
            return this.bLexicon.numberOfEntries();
        }

        @Override
        public LexiconEntry getLexiconEntry(String term) {
            LexiconEntry rtr = this.pLexicon.getLexiconEntry(term);
            if (rtr == null)
                return null;
            LexiconEntry background = this.bLexicon.getLexiconEntry(term);
            if (background != null)
                rtr.add(background);
            return rtr;
        }

        @Override
        public Map.Entry<String,LexiconEntry> getLexiconEntry(int termid) {
            Map.Entry<String,LexiconEntry> rtr = this.pLexicon.getLexiconEntry(termid);
            if (rtr == null)
                return null;
            LexiconEntry background = this.bLexicon.getLexiconEntry(rtr.getKey());
            rtr.getValue().add(background);
            return rtr;
        }

        @Override
        public Map.Entry<String,LexiconEntry> getIthLexiconEntry(int index) {
            Map.Entry<String,LexiconEntry> rtr = this.pLexicon.getIthLexiconEntry(index);
            if (rtr == null)
                return null;
            LexiconEntry background = this.bLexicon.getLexiconEntry(rtr.getKey());
            rtr.getValue().add(background);
            return rtr;
        }

        @Override
        public Iterator<Map.Entry<String,LexiconEntry>> getLexiconEntryRange(String from, String to) {
            throw new UnsupportedOperationException("Not yet implemented");
        }

        @Override
        public Iterator<Map.Entry<String,LexiconEntry>> iterator() {
            throw new UnsupportedOperationException("Not yet implemented");
        }

        @Override
        public void close() throws IOException {
            this.pLexicon.close();
        }
    }


}