package core.adapters;

import java.net.URL;
import weka.classifiers.functions.CoevolutionaryRuleExtractor;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class TrainAndTestInstances {

    private String baseFolder = "/monks/";
    private String suffix = ".arff";
    private Instances trainData;
    private Instances testData;
    String trainpath;
    String testpath;
    private MyNiceInterface myinterface = new MyNiceInterface();

    public TrainAndTestInstances(String name) {
        this(name, name);
    }

    public TrainAndTestInstances(String trainname, String testname) {
        String trainPathString = null;
        try {
            trainPathString = baseFolder + trainname + suffix;
            String testPathString = baseFolder + testname + suffix;
            URL train = CoevolutionaryRuleExtractor.class.getResource(trainPathString);
            URL test = CoevolutionaryRuleExtractor.class.getResource(testPathString);
            trainpath = train.getPath();
            trainData = DataSource.read(trainpath);
            testpath = test.getPath();
            testData = DataSource.read(testpath);
            trainData.setClassIndex(trainData.numAttributes() - 1);
            testData.setClassIndex(testData.numAttributes() - 1);
        } catch (Exception ex) {
            System.out.println("trainPathString = " + trainPathString);
            System.out.println("trainpath = " + trainpath);
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

    public MyNiceInterface paths() {
        return myinterface;
    }

    public class MyNiceInterface {

        public String train() {
            return trainpath;
        }

        public String test() {
            return testpath;
        }
    }
}
