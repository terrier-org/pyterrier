package org.terrier.python;
import org.terrier.matching.models.WeightingModel;
public class TestModel extends WeightingModel {
    public TestModel() {}
    public final String getInfo() { return "TestModel"; }
    public final double score(double tf, double docLength) { return tf/ super.numberOfDocuments; }	
}