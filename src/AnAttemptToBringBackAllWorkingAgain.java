
import core.adapters.TrainAndTestInstances;
import core.adapters.DataAdapter;
import weka.classifiers.functions.CoevolutionCallback;
import core.copop.CoPopulations;
import core.ga.ops.ec.ExecutionEnv;
import core.utils.ui.MMMGraph;
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
            TrainAndTestInstances data = new TrainAndTestInstances("iris");
            CoevolutionaryRuleExtractor classifier = new CoevolutionaryRuleExtractor();
            final MMMGraph graphR = new MMMGraph("Rules");
            final MMMGraph graphRS = new MMMGraph("RulesSets");
            DataAdapter adapter = new DataAdapter(data.train());
            ExecutionEnv ec = CoevolutionaryRuleExtractor.constructEnvironmentForWEKAInstances(adapter);
            EvoElitistSelection.constructGUI(graphR, graphRS, ec);
            classifier.setCallback(new CoevolutionCallback() {

                public void coevolutionCallback(CoPopulations pops) {
                    graphR.add(pops.ruleStats());
                    graphRS.add(pops.ruleSetStats());
                }
            });
//            classifier.buildClassifier(data.train());

            //Evaluation eval = new Evaluation(data.test());
            //eval.evaluateModel(classifier, data.test());
            String s = Evaluation.evaluateModel(classifier, new String[]{
                        "-i",
                        "-t", data.paths().train(),
                        "-T", data.paths().test()
                    });
            System.out.println(s);
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }
}
