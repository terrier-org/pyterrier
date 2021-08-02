package org.terrier.python;

import org.terrier.matching.models.WeightingModel;
import org.terrier.matching.MatchingQueryTerms;
import org.terrier.matching.MatchingQueryTerms.MatchingTerm;
import org.terrier.querying.Manager;
import org.terrier.querying.Process;
import org.terrier.querying.ProcessPhaseRequisites;
import org.terrier.querying.ManagerRequisite;
import org.terrier.querying.Request;

@ProcessPhaseRequisites(ManagerRequisite.MQT)
/** Takes an object from the SearchRequest's context mapping, and uses that as the weighting model. */
public class WmodelFromContextProcess implements Process {

    public void process​(Manager manager, Request rq) {
        Object _wmodel = rq.getContextObject​("context_wmodel");
        System.out.println(_wmodel.getClass().getName());
        if (_wmodel == null) {
            throw new IllegalStateException("WmodelFromContextProcess invoked, but no context object found under key 'context_wmodel'");
        }
        WeightingModel wmodel = (WeightingModel) _wmodel;
        MatchingQueryTerms mqt = rq.getMatchingQueryTerms();
        for(MatchingTerm e : mqt)
		{
            if ( e.getValue().termModels.size() > 0)
            {
                e.getValue().termModels.set(0, wmodel.clone());
            } else {
                e.getValue().termModels.add(wmodel.clone());
            }
        }
    }
}