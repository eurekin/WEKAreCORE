
import java.io.File;
import java.net.URL;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.CoevolutionaryRuleExtractor;

/**
 *
 * @author gmatoga
 */
public class AnAttemptToBringBackAllWorkingAgain {

    public static void main(String[] args) {
        System.out.println("Attempt to bring back all working again");
        func();
    }

    public static void func() {
        try {
            String[] args = new String[]{};
            //System.out.println("args = " + Arrays.deepToString(args));
            //runClassifier(new CoevolutionaryRuleExtractor(), args);
            TrainAndTestInstances data = new TrainAndTestInstances("iris");
            Classifier classifier = new CoevolutionaryRuleExtractor();
            File f = new File("src/monks/iris.arff");
            System.out.println("f.exists? " + f.exists());
            URL resource = CoevolutionaryRuleExtractor.class.getResource("/monks/iris.arff");


            classifier.buildClassifier(data.train());
            Evaluation eval = new Evaluation(data.test());
            eval.evaluateModel(classifier, data.test());
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }
}
