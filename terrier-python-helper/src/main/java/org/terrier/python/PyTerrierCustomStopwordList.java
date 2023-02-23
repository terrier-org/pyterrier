package org.terrier.python;

import org.terrier.utility.ApplicationSetup;
import org.terrier.terms.TermPipeline;
import org.terrier.structures.Index;
import org.terrier.structures.IndexConfigurable;
import org.terrier.structures.PropertiesIndex;
import java.util.Set;
import java.util.HashSet;
import java.util.Properties;


public abstract class PyTerrierCustomStopwordList implements TermPipeline
{
    public static String PROPERTY = "pyterrier.stopwords";
    public static class Retrieval extends PyTerrierCustomStopwordList implements IndexConfigurable {

        boolean configured = false;

        public Retrieval(TermPipeline _next) {
            super(_next);
        }
        
        public Retrieval() {
            super();
        }
        
        public void setIndex(Index index) {
            if (! (index instanceof PropertiesIndex)) {
                throw new RuntimeException("Can only use with PropertiesIndex - other index types such as memory not supported");
            }
            PropertiesIndex pi = (PropertiesIndex)index;
            this.setup(pi.getProperties());
            configured = true;
        }

        public void processTerm(final String t)
	    {
            if (! configured) {
                throw new IllegalStateException("PyTerrierCustomStopwordList$Retrieval used, but index has not yet been configured");
            }
            super.processTerm(t);
        }
    }

    public static class Indexing extends PyTerrierCustomStopwordList {

        public Indexing(TermPipeline _next) {
            super(_next);
            setup(ApplicationSetup.getProperties());
        }
        
        public Indexing() {
            super();
            setup(ApplicationSetup.getProperties());
        }
    }

    TermPipeline next;
    Set<String> stopWords = new HashSet<>();
    
    public PyTerrierCustomStopwordList(TermPipeline _next) {
        next = _next;
    }
    
    public PyTerrierCustomStopwordList() {
        next = null;
    }

    protected void setup(Properties props) {
        String termString = props.getProperty(PROPERTY, null);
        if (termString == null) {
            return;
        }
        String[] terms = termString.split(",");
        for (String t : terms) {
            stopWords.add(t);
        }
    } 

    public boolean isStopword(String t) {
        return stopWords.contains(t);
    }

    /** 
	 * Checks to see if term t is a stopword. If so, then the TermPipeline
	 * is exited. Otherwise, the term is passed on to the next TermPipeline
	 * object. This is the TermPipeline implementation part of this object.
	 * @param t The term to be checked.
	 */
	public void processTerm(final String t)
	{
		if (stopWords.contains(t))
			return;
		next.processTerm(t);
	}
	
	/** {@inheritDoc} */
	public boolean reset() {
		return next.reset();
	}
}