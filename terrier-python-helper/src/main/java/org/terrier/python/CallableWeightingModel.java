package org.terrier.python;
import org.terrier.matching.models.WeightingModel;
import org.terrier.structures.postings.Posting;
import org.terrier.structures.EntryStatistics;
import org.terrier.structures.CollectionStatistics;
import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectStreamException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/** A weighting model class that includes a Callback interface that can be implemented in Python.
 */
public class CallableWeightingModel extends WeightingModel {

    public static interface Callback extends Externalizable {
        public double score(double keyFrequency, Posting posting, EntryStatistics entryStats, CollectionStatistics collStats);
        public default double score1(Posting posting, EntryStatistics entryStats, CollectionStatistics collStats) {
            return score(1.0d, posting, entryStats, collStats);
        }
    }

    Callback scoringClass;

    private CallableWeightingModel() {}

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

    private void writeObject(java.io.ObjectOutputStream out) throws IOException {
        scoringClass.writeExternal(out);
    }
    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        scoringClass.readExternal(in);
    }
    private void readObjectNoData() throws ObjectStreamException {}
}