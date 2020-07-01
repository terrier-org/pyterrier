package org.terrier.python;

import org.terrier.matching.Matching;
import org.terrier.matching.QueryResultSet;
import org.terrier.matching.MatchingQueryTerms;
import org.terrier.matching.ResultSet;
import org.terrier.structures.CollectionStatistics;
import org.terrier.querying.SearchRequest;
import org.terrier.structures.MetaIndex;
import org.terrier.structures.Index;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RequestContextMatching implements Matching {

    protected static final Logger logger = LoggerFactory.getLogger(RequestContextMatching.class);

    public static String CONTROL_META = "request_context_matching";
    public static String CONTEXT_SOURCE = "request_context_matching_source";
    public static String CONTEXT_SCORES = "request_context_matching_scores";

    public static Factory of(SearchRequest srq) {
        return new Factory(srq);
    }

    public static class Factory {
        SearchRequest srq;

        private Factory(SearchRequest srq) {
            this.srq = srq;
            srq.setControl("matching", RequestContextMatching.class.getName());
        }

        public Factory fromDocids(int[] docids) {
            srq.setControl(CONTROL_META, "docids");
            srq.setContextObject(CONTEXT_SOURCE, docids);
            return this;
        }

        public Factory fromDocnos(String[] docs) {
            srq.setControl(CONTROL_META, "docnos");
            srq.setContextObject(CONTEXT_SOURCE, docs);
            return this;
        }

        public Factory withScores(double[] scores) {
            srq.setContextObject(CONTEXT_SCORES, scores);
            return this;
        }

        public SearchRequest build() {
            return this.srq;
        }
    }

    Index index;
    public RequestContextMatching(Index index){
        this.index = index;
    }
     
    public ResultSet match(String queryNumber, MatchingQueryTerms queryTerms) {
        
        try{
            String source = queryTerms.getRequest().getControl(CONTROL_META);
            Object o_docs = queryTerms.getRequest().getContextObject(CONTEXT_SOURCE);
            Object o_scores = queryTerms.getRequest().getContextObject(CONTEXT_SCORES);


            int[] docids;
            if ("docids".equals(source) || "docid".equals(source))
            {
                docids = (int[])o_docs;
            }
            else
            {
                Index _index = this.index;
                if (queryTerms.getRequest() != null && queryTerms.getRequest().getIndex() != null)
                {
                    _index = queryTerms.getRequest().getIndex();
                }
                MetaIndex meta = _index.getMetaIndex();
                String[] sourceMeta = (String[])o_docs;
                docids = new int[sourceMeta.length];
                for(int i=0;i<docids.length;i++)
                {
                    docids[i] = meta.getDocument(source, sourceMeta[i]);
                }
            }
            double[] scores;
            if (o_scores != null)
            {
                scores = (double[])o_scores;
            } else {
                scores = new double[docids.length];
            }
            logger.info("Found " + docids.length + " documents from Request for query " + queryNumber);
            return new QueryResultSet(docids, scores, new short[docids.length]);
        } catch (Exception e) {
            throw new RuntimeException("Problem making resultset", e);
        }
    }

    public void setCollectionStatistics(CollectionStatistics cs) {

    }

    public String getInfo() {
        return this.getClass().getSimpleName();
    }
}