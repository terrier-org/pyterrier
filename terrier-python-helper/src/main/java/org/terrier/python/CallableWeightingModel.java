package org.terrier.python;
import org.terrier.matching.models.WeightingModel;
import org.terrier.structures.postings.Posting;
import org.terrier.structures.EntryStatistics;
import org.terrier.structures.CollectionStatistics;

/** A weighting model class that includes a Callback interface that can be implemented in Python.
 */
public class CallableWeightingModel extends WeightingModel {

    public static interface Callback {
        public double score(double keyFrequency, Posting posting, EntryStatistics entryStats, CollectionStatistics collStats);
        public default double score1(Posting posting, EntryStatistics entryStats, CollectionStatistics collStats) {
            return score(1.0d, posting, entryStats, collStats);
        }
    }

    final Callback scoringClass;

    public CallableWeightingModel(Callback _scoringClass) {
        scoringClass = _scoringClass;
    }

    @Override
    public double score(Posting p) {
        return scoringClass.score(super.keyFrequency, p, super.es, super.cs);
    }
    
    @Override
    public double score(double a, double  b) {
        throw new UnsupportedOperationException();
    }

    @Override
    public String getInfo() {
        return this.getClass().getSimpleName();
    }
}