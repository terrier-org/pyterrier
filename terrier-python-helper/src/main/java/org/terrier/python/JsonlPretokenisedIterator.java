package org.terrier.python;

import java.util.Iterator;
import java.util.HashMap;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Scanner;
import java.io.File;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.lang.IllegalStateException;
import org.terrier.structures.indexing.DocumentPostingList;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

/** This class allows reading FlatJSONDocument objects, line by line from a file */
public class JsonlPretokenisedIterator implements Iterator<Map.Entry<Map<String,String>, DocumentPostingList>> {
    String path;
    Scanner scanner;

    public JsonlPretokenisedIterator(String path) {
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
    @SuppressWarnings("unchecked")
    public Map.Entry<Map<String,String>, DocumentPostingList> next() throws NoSuchElementException {
        try {
            String rawJson = this.scanner.nextLine().trim();
            ObjectMapper mapper = new ObjectMapper();
			Map<String, Object> nestedMap = (Map<String,Object>) mapper.readValue(rawJson, Map.class);
            DocumentPostingList pl = new DocumentPostingList();
            Map<String,String> properties = new HashMap<>();
            nestedMap.entrySet().stream()
                .filter( entry -> ! entry.getKey().equals("toks") )
                .forEach( entry -> properties.put( entry.getKey(), (String)entry.getValue()) );
            Map<String, Integer> toks = (Map<String, Integer>) nestedMap.get("toks");
            toks.forEach( (t, freq) -> pl.insert(freq, t) );
            return new org.terrier.structures.collections.MapEntry<Map<String,String>,DocumentPostingList>(properties, pl);

        } catch (IllegalStateException ex) {
            throw new NoSuchElementException("Illegal state");
        } catch (JsonProcessingException ex) {
            throw new NoSuchElementException("Invalid JSON");
        } catch (IOException ex) {
            throw new NoSuchElementException("IO Error");
        }
    }
}