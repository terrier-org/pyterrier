package org.terrier.python;
import org.terrier.matching.models.WeightingModel;
public class TestModel {

    public static class Constant extends WeightingModel {
        public Constant() {}
        public final String getInfo() { return "Constant"; }
        public final double score(double tf, double docLength) { return 1; }
    }

    public static class TFOverN extends WeightingModel {
        public TFOverN() {}
        public final String getInfo() { return "TFOverN"; }
        public final double score(double tf, double docLength) { return tf/ super.numberOfDocuments; }
    }

    public static class F extends WeightingModel {
        public F() {}
        public final String getInfo() { return "F"; }
        public final double score(double tf, double docLength) { return super.termFrequency; }
    }

    public static class Nt extends WeightingModel {
        public Nt() {}
        public final String getInfo() { return "Nt"; }
        public final double score(double tf, double docLength) { return super.documentFrequency; }
    }
}