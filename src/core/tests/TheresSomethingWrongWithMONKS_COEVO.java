package core.tests;

import core.adapters.TrainAndTestInstances;
import core.copop.CoPopulations;
import core.ui.evoGUI;
import core.utils.ui.MMMGraph;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.CoevolutionCallback;
import weka.classifiers.functions.CoevolutionaryRuleExtractor;

/**
 *
 * @author gmatoga
 */
public class TheresSomethingWrongWithMONKS_COEVO {

    String options = "-D";
    TrainAndTestInstances data;
    MMMGraph graphR = new MMMGraph("rules");
    MMMGraph graphRS = new MMMGraph("rulesets");
    CoevolutionaryRuleExtractor core = new CoevolutionaryRuleExtractor();
    private CoevolutionCallback callback = new MyCallback();

    public TheresSomethingWrongWithMONKS_COEVO() {
        evoGUI evoGUI = new evoGUI(graphR, graphRS);
//        String file = "monks-2.train";
//        String file = "soybean";
        String file = "iris";
        data = new TrainAndTestInstances(file);
    }

    private void prepare() {
        core.setCallback(callback);
        core.setGenerations(30000);
        core.setRuleSetMutationProbability(0.02);
        core.setRuleMutationProbability(0.02);
        core.setTokenCompetitionEnabled(false);
        core.setMaxRulesCount(10);
        core.setRulePopulationSize(200);
        core.setRuleSetPopulationSize(200);
        core.setDebug(false);
    }

    private void run() throws Exception {
        prepare();
        core.buildClassifier(data.train());
        Evaluation eval = new Evaluation(data.train());
        eval.evaluateModel(core, data.train());
        System.out.println("accuracy : " + eval.pctCorrect());
    }

    public static void main(String[] args) throws Exception {
        TheresSomethingWrongWithMONKS_COEVO test = new TheresSomethingWrongWithMONKS_COEVO();
        test.run();
    }

    public class MyCallback implements CoevolutionCallback {

        public void coevolutionCallback(CoPopulations pops) {
            graphR.add(pops.ruleStats());
            graphRS.add(pops.ruleSetStats());
        }
    }
}
