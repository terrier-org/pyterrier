package org.terrier.python;

import java.util.Iterator;
import org.terrier.indexing.Collection;
import org.terrier.indexing.Document;

/** This class allows easier creating Collection objects from iterators */
public class CollectionFromDocumentIterator implements Collection
{
    Iterator<Document> docIter;
    Document currentDoc;
    
    public CollectionFromDocumentIterator(Iterator<Document> _docIter) {
        this.docIter = _docIter;
    }

    public CollectionFromDocumentIterator(Iterable<Document> _docIterable) {
        this.docIter = _docIterable.iterator();
    }

    @Override
    public boolean nextDocument() {
        if (! docIter.hasNext())
        {
            return false;
        }
        currentDoc = docIter.next();
        return true;
    }

    @Override
    public Document getDocument() {
        return currentDoc;
    }

    @Override
    public boolean endOfCollection() {
        return ! docIter.hasNext();
    }

    @Override
    public void reset() {
        throw new UnsupportedOperationException();
    }
    
    @Override
    public void close() {}
}