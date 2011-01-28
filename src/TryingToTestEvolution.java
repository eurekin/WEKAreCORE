
import core.adapters.TrainAndTestInstances;
import core.adapters.DataAdapter;
import core.evo.EvolutionPopulation;
import core.ga.ops.ec.ExecutionEnv;
import core.utils.ui.MMMGraph;
import java.awt.HeadlessException;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.CoevolutionaryRuleExtractor;
import weka.classifiers.functions.EvolutionCallback;
import weka.classifiers.functions.EvolutionaryRuleExtractor;

/**
 *
 * @author gmatoga
 */
public class TryingToTestEvolution {

    public static void main(String[] args) {
        System.out.println(
                "There's some nasty bug with Evolution, Diabetes dataset"
                + " and distributed experiment. Together it's a perfect case for"
                + "out of memory exception to be raised. here I'm trying"
                + "to reproduce it.");
        for (int i = 0; i < 100; i++) {
            func();

        }
    }

    public static void func() {
        try {
            TrainAndTestInstances data = new TrainAndTestInstances("diabetes");
            EvolutionaryRuleExtractor classifier = new EvolutionaryRuleExtractor();
            classifier.setGenerations(12);
            DataAdapter adapter = new DataAdapter(data.train());
            ExecutionEnv ec = CoevolutionaryRuleExtractor.constructEnvironmentForWEKAInstances(adapter);
//            evoGUI(ec, classifier);
            String s = Evaluation.evaluateModel(classifier, new String[]{
                        "-i", "-t", data.paths().train(),
                        "-T", data.paths().test()
                    });
            System.out.println(s);
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }

    private static void evoGUI(ExecutionEnv ec, EvolutionaryRuleExtractor classifier) throws HeadlessException {
        final MMMGraph graphR = new MMMGraph("Rules");
        final MMMGraph graphRS = new MMMGraph("RulesSets");
        EvoElitistSelection.constructGUI(graphR, graphRS, ec);
        classifier.setCallback(new EvolutionCallback() {

            public void coevolutionCallback(EvolutionPopulation pop) {
                graphRS.add(pop.stats());
            }
        });
    }
}
