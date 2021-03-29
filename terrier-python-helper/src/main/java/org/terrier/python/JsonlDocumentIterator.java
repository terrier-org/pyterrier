package org.terrier.python;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Scanner;
import java.io.File;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.lang.IllegalStateException;
import org.terrier.indexing.FlatJSONDocument;
import com.fasterxml.jackson.core.JsonProcessingException;

/** This class allows reading FlatJSONDocument objects, line by line from a file */
public class JsonlDocumentIterator implements Iterator<FlatJSONDocument> {
    String path;
    Scanner scanner;

    public JsonlDocumentIterator(String path) {
        this.path = path;
        this.scanner = null;
    }

    @Override
    public boolean hasNext() {
        try {
            if (this.scanner == null) {
                this.scanner = new Scanner(new File(this.path));
            }
            return this.scanner.hasNextLine();
        } catch (IllegalStateException ex) {
            return false; // scanner complete / in otherwise invalid state
        } catch (FileNotFoundException ex) {
            return false;
        }
    }

    @Override
    public FlatJSONDocument next() throws NoSuchElementException {
        try {
            String line = this.scanner.nextLine().trim();
            return new FlatJSONDocument(line);
        } catch (IllegalStateException ex) {
            throw new NoSuchElementException("Illegal state");
        } catch (JsonProcessingException ex) {
            throw new NoSuchElementException("Invalid JSON");
        } catch (IOException ex) {
            throw new NoSuchElementException("IO Error");
        }
    }
}