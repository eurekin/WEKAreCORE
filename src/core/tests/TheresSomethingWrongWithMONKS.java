package core.tests;

import core.adapters.TrainAndTestInstances;
import core.evo.EvolutionPopulation;
import core.ui.evoGUI;
import core.utils.ui.MMMGraph;
import weka.classifiers.functions.EvolutionCallback;
import weka.classifiers.functions.EvolutionaryRuleExtractor;

/**
 *
 * @author gmatoga
 */
public class TheresSomethingWrongWithMONKS {

    String options = "-D";
    MMMGraph graphR = new MMMGraph("rules");
    MMMGraph graphRS = new MMMGraph("rulesets");
    EvolutionaryRuleExtractor evol = new EvolutionaryRuleExtractor();
    TrainAndTestInstances data;
    private EvolutionCallback callback = new MyEvoBack();

    public TheresSomethingWrongWithMONKS() {
        evoGUI evoGUI = new evoGUI(graphR, graphRS);
        String file = "monks-2.train";
//        String file = "soybean";
//        String file = "iris";
        data = new TrainAndTestInstances(file);
    }

    private void prepare() {
        evol.setCallback(callback);
        evol.setGenerations(30000);
        evol.setRuleSetMutationProbability(0.01);
        evol.setRuleMutationProbability(0.01);
        evol.setTokenCompetitionEnabled(false);
        evol.setMaxRulesCount(10);
        evol.setRuleSetPopulationSize(200);
        evol.setDebug(false);
    }

    private void run() throws Exception {
        prepare();
        evol.buildClassifier(data.train());
    }

    public static void main(String[] args) throws Exception {
        TheresSomethingWrongWithMONKS test = new TheresSomethingWrongWithMONKS();
        test.run();
    }

    public class MyEvoBack implements EvolutionCallback {

        public void coevolutionCallback(EvolutionPopulation pop) {
            graphRS.add(pop.stats());
        }
    }
}