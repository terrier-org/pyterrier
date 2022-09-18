/*
 * Terrier - Terabyte Retriever 
 * Webpage: http://terrier.org 
 * Contact: terrier{a.}dcs.gla.ac.uk
 * University of Glasgow - School of Computing Science
 * http://www.gla.ac.uk/
 * 
 * The contents of this file are subject to the Mozilla Public License
 * Version 1.1 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 * http://www.mozilla.org/MPL/
 *
 * Software distributed under the License is distributed on an "AS IS"
 * basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See
 * the License for the specific language governing rights and limitations
 * under the License.
 *
 * The Original Code is MemoryIndexer.java.
 *
 * The Original Code is Copyright (C) 2004-2020 the University of Glasgow.
 * All Rights Reserved.
 *
 * Contributor(s):
 *   Richard McCreadie <richard.mccreadie@glasgow.ac.uk>
 *   Stuart Mackie <s.mackie.1@research.gla.ac.uk>
 */
package org.terrier.python;

import gnu.trove.TIntHashSet;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import org.terrier.indexing.Collection;
import org.terrier.indexing.Document;
import org.terrier.realtime.memory.MemoryIndex;
import org.terrier.realtime.memory.fields.MemoryFieldsIndex;
import org.terrier.structures.Index;
import org.terrier.structures.indexing.DocumentPostingList;
import org.terrier.structures.indexing.FieldDocumentPostingList;
import org.terrier.structures.indexing.Indexer;
import org.terrier.terms.TermPipeline;
import org.terrier.utility.FieldScore;

/**
 * FIXME
 */
public class MemoryIndexer extends Indexer {

	/**
	 * Constructor.
	 */
	public MemoryIndexer() {
        this(false);

	}
	
	public MemoryIndexer(boolean fields) {
		if (fields) {
			memIndex = new MemoryFieldsIndex();
		} else memIndex = new MemoryIndex();
		if (this.getClass() == MemoryIndexer.class)
            init();
	}

	/** FIXME */
	class BasicTermProcessor implements TermPipeline {
		public void processTerm(String term) {
			if (term != null) {
				termsInDocument.insert(term);
				numOfTokensInDocument++;
			}
		}

		public boolean reset() {
			return true;
		}
	}
	
	/** FIXME */
	class FieldTermProcessor implements TermPipeline {
		final TIntHashSet fields = new TIntHashSet(numFields);
		final boolean ELSE_ENABLED = fieldNames.containsKey("ELSE");
		final int ELSE_FIELD_ID = fieldNames.get("ELSE") -1;
		public void processTerm(String term)
		{
			
			/* null means the term has been filtered out (eg stopwords) */
			if (term != null)
			{
				/* add term to Document tree */
				for (String fieldName: termFields)
				{
					int tmp = fieldNames.get(fieldName);
					if (tmp > 0)
					{
						fields.add(tmp -1);
					}
				}
				if (ELSE_ENABLED && fields.size() == 0)
				{
					fields.add(ELSE_FIELD_ID);
				}
				((FieldDocumentPostingList)termsInDocument).insert(term,fields.toArray());
				numOfTokensInDocument++;
				fields.clear();
			}
		}
		@Override
		public boolean reset() {
			return true;
		}
	}

	/** FIXME */
	private MemoryIndex memIndex;

	/** FIXME */
	private DocumentPostingList termsInDocument;

	/** FIXME */
	private Set<String> termFields;

	/** FIXME */
	private int numOfTokensInDocument = 0;

	/** FIXME */
    private int numberOfDocuments;

	@Override
	public void indexDocuments(Iterator<Map.Entry<Map<String,String>,DocumentPostingList>> iterDocs)
	{
		while(iterDocs.hasNext()) {
			Map.Entry<java.util.Map<String,String>,DocumentPostingList> d = iterDocs.next();
			if (d == null)
				continue;
			try {
				memIndex.indexDocument(d.getKey(), d.getValue());
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
    
	@Override
    public void index(Collection collection) {
        this.createDirectIndex(collection);
		this.createInvertedIndex();
    }
	
	/** FIXME */
	public void createDirectIndex(Collection collection) {
		numFields = FieldScore.FIELDS_COUNT;
		
		long startCollection = System.currentTimeMillis();
		boolean notLastDoc = false;
		while ((notLastDoc = collection.nextDocument())) {
			Document doc = collection.getDocument();
			if (doc == null)
				continue;
			numberOfDocuments++;
			createDocumentPostings();
			String term;
			while (!doc.endOfDocument()) {
				if ((term = doc.getNextTerm()) != null && !term.equals("")) {
					termFields = doc.getFields();
					pipeline_first.processTerm(term);
					
				}
				if (MAX_TOKENS_IN_DOCUMENT > 0
						&& numOfTokensInDocument > MAX_TOKENS_IN_DOCUMENT)
					break;
			}
			while (!doc.endOfDocument())
				doc.getNextTerm();
			pipeline_first.reset();
			try {
				memIndex.indexDocument(doc.getAllProperties(), termsInDocument);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		if (!notLastDoc) {
			try {
				collection.close();
			} catch (IOException e) {
				logger.warn("Couldnt close collection", e);
			}
		}
		long endCollection = System.currentTimeMillis();
		long secs = ((endCollection - startCollection) / 1000);
		logger.info("Collection took " + secs
				+ " seconds to index " + "(" + numberOfDocuments
				+ " documents)");
		if (secs > 3600)
			logger.info("Rate: "
					+ ((double) numberOfDocuments / ((double) secs / 3600.0d))
					+ " docs/hour");

	}

	/** FIXME */
	public void createInvertedIndex() {
	}

	/** FIXME */
	protected TermPipeline getEndOfPipeline() {
		if (fieldNames.size()>0) {
			return new FieldTermProcessor();
		}
		else {
			return new BasicTermProcessor();
		}
	}

	/** FIXME */
	void createDocumentPostings() {
		if (numFields>0) termsInDocument = new FieldDocumentPostingList(numFields);
		else termsInDocument = new DocumentPostingList();
	}

	public Index getIndex() {
		return memIndex;
	}

}
