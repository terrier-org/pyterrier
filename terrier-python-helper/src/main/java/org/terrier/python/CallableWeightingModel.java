package org.terrier.python;
import org.terrier.matching.models.WeightingModel;
import org.terrier.structures.postings.Posting;
import org.terrier.structures.EntryStatistics;
import org.terrier.structures.CollectionStatistics;
import java.io.IOException;
import java.io.ObjectStreamException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.ByteBuffer;

/** A weighting model class that includes a Callback interface that can be implemented in Python.
 */
public class CallableWeightingModel extends WeightingModel {

    public static interface Callback {
        public double score(double keyFrequency, Posting posting, EntryStatistics entryStats, CollectionStatistics collStats);
        public default double score1(Posting posting, EntryStatistics entryStats, CollectionStatistics collStats) {
            return score(1.0d, posting, entryStats, collStats);
        }
        public ByteBuffer serializeFn();
    }

    public Callback scoringClass;

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

    private void writeObject(ObjectOutputStream out) throws IOException {}
    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {}
    private void readObjectNoData() throws ObjectStreamException {}
}