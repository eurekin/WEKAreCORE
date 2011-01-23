
import java.net.URL;
import weka.classifiers.functions.CoevolutionaryRuleExtractor;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class TrainAndTestInstances {

    private String baseFolder = "/monks/";
    private String suffix = ".arff";
    private Instances trainData;
    private Instances testData;

    public TrainAndTestInstances(String name) {
        try {
            String trainPathString = baseFolder + name + suffix;
            String testPathString = baseFolder + name + suffix;
            URL train = CoevolutionaryRuleExtractor.class.getResource(trainPathString);
            URL test = CoevolutionaryRuleExtractor.class.getResource(testPathString);
            trainData = DataSource.read(train.getPath());
            testData = DataSource.read(test.getPath());
            trainData.setClassIndex(trainData.numAttributes() - 1);
            testData.setClassIndex(testData.numAttributes() - 1);
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }

    public Instances train() {
        return trainData;
    }

    public void setTrainData(Instances trainData) {
        this.trainData = trainData;
    }

    public Instances test() {
        return testData;
    }

    public void setTestData(Instances testData) {
        this.testData = testData;
    }
}
