package core.tests;

import core.adapters.TrainAndTestInstances;
import core.copop.CoPopulations;
import core.evo.EvolutionPopulation;
import core.ui.evoGUI;
import core.utils.ui.MMMGraph;
import java.awt.HeadlessException;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.CoevolutionCallback;
import weka.classifiers.functions.CoevolutionaryRuleExtractor;
import weka.classifiers.functions.EvolutionCallback;
import weka.classifiers.functions.EvolutionaryRuleExtractor;

/**
 *
 * @author gmatoga
 */
public class EvoVsCoevo {

    static int generations = 1000;
    static double rmp = 0.01;
    static double rsmp = 0.01;
    private static int rules = 5;

    public static void main(String[] args) throws Exception {
        new evoGUI(new MMMGraph("ignore"), graphRS);
        new evoGUI(graphR, graphRS2);
        final TrainAndTestInstances data = new TrainAndTestInstances("monks-2.train");
        ScheduledExecutorService exs = Executors.newScheduledThreadPool(4);
        CompletionService<Evaluation> pool =
                new ExecutorCompletionService<Evaluation>(exs);
        int howMany = 100;
        for (int i = 0; i < howMany; i++) {
            submit(pool, data, evo());
            submit(pool, data, coev());
        }
        Evaluation get;
        for (int i = 0; i < howMany; i++) {
            get = pool.take().get();
            System.out.println(get.pctCorrect());

        }
    }

    private static void submit(CompletionService pool,
            final TrainAndTestInstances data, final Classifier classifier) {
        pool.submit(new Callable<Evaluation>() {

            public Evaluation call() throws Exception {
                return func(classifier, data);
            }
        });
    }

    public static Evaluation func(Classifier classifier, TrainAndTestInstances data) {
        try {
            Evaluation eval = new Evaluation(data.train());
            classifier.buildClassifier(data.train());
            eval.evaluateModel(classifier, data.test());
            return eval;
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }

    private static EvolutionaryRuleExtractor evo() throws HeadlessException {
        EvolutionaryRuleExtractor classifier = new EvolutionaryRuleExtractor();
        classifier.setGenerations(generations);
        classifier.setRuleMutationProbability(rmp);
        classifier.setRuleSetMutationProbability(rsmp);
        classifier.setMaxRulesCount(rules);
        evoGUI(classifier);
        return classifier;
    }

    private static CoevolutionaryRuleExtractor coev() throws HeadlessException {
        CoevolutionaryRuleExtractor classifier = new CoevolutionaryRuleExtractor();
        classifier.setGenerations(generations);
        classifier.setRuleMutationProbability(rmp);
        classifier.setRuleSetMutationProbability(rsmp);
        classifier.setMaxRulesCount(rules);
        coevoGUI(classifier);
        return classifier;
    }

    private static void evoGUI(EvolutionaryRuleExtractor classifier) throws HeadlessException {
        classifier.setCallback(new EvolutionCallbackImpl(graphRS));
    }

    private static void coevoGUI(CoevolutionaryRuleExtractor classifier) throws HeadlessException {
        classifier.setCallback(new CoevolutionCallback() {

            public void coevolutionCallback(CoPopulations pops) {
                graphRS2.add(pops.ruleSetStats());
                graphR.add(pops.ruleStats());
            }
        });
    }
    static final MMMGraph graphR = new MMMGraph("Rules");
    static final MMMGraph graphRS = new MMMGraph("RulesSets");
    static final MMMGraph graphRS2 = new MMMGraph("RulesSets");

    private static class EvolutionCallbackImpl implements EvolutionCallback {

        private final MMMGraph graphRS;
        private final static Object[] lock = new Object[0];

        public EvolutionCallbackImpl(MMMGraph graphRS) {
            synchronized (lock) {
                this.graphRS = graphRS;
            }
        }

        public void coevolutionCallback(EvolutionPopulation pop) {
            graphRS.add(pop.stats());
        }
    }
}
