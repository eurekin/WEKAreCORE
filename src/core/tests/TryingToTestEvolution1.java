package core.tests;

import core.adapters.TrainAndTestInstances;
import core.adapters.DataAdapter;
import core.copop.CoPopulations;
import core.evo.EvolutionPopulation;
import core.ga.ops.ec.ExecutionEnv;
import core.ui.evoGUI;
import core.utils.ui.MMMGraph;
import java.awt.HeadlessException;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.processing.Completions;
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
public class TryingToTestEvolution1 {

    public static void main(String[] args) {
        System.out.println(
                "There's some nasty bug with Evolution, Diabetes dataset"
                + " and distributed experiment. Together it's a perfect case for"
                + "out of memory exception to be raised. here I'm trying"
                + "to reproduce it.");
        evoGUI evoGUI = new evoGUI(new MMMGraph("ignore"), graphRS);
        evoGUI evoGUI2 = new evoGUI(graphR, graphRS2);
        final TrainAndTestInstances data = new TrainAndTestInstances("diabetes");
        ExecutorService pool = Executors.newScheduledThreadPool(2);
        int howMany = 100;
        for (int i = 0; i < howMany; i++) {
            submit(pool, data);
            pool.submit(
                    new Callable<String>() {

                        public String call() throws Exception {
                            return func(coev(data), data);
                        }
                    });
        }
        pool.shutdown();
        CompletionService<String> finish = new ExecutorCompletionService<String>(pool);
        for (int i = 0; i < howMany; i++) {
            try {
                String get = finish.take().get();
                System.out.println(get);
            } catch (InterruptedException ex) {
                Logger.getLogger(TryingToTestEvolution1.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(TryingToTestEvolution1.class.getName()).log(Level.SEVERE, null, ex);
            }

        }
    }

    private static void submit(ExecutorService pool, final TrainAndTestInstances data) {
        pool.submit(new Callable<String>() {

            public String call() throws Exception {
                return func(evo(data), data);
            }
        });
    }

    public static String func(Classifier classifier, TrainAndTestInstances data) {
        try {
            String s = Evaluation.evaluateModel(classifier, new String[]{
                        "-i", "-t", data.paths().train(),
                        "-T", data.paths().test()
                    });
            return s;
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }
    static int generations = 120;

    private static EvolutionaryRuleExtractor evo(TrainAndTestInstances data) throws HeadlessException {
        EvolutionaryRuleExtractor classifier = new EvolutionaryRuleExtractor();
        classifier.setGenerations(generations);
        DataAdapter adapter = new DataAdapter(data.train());
        ExecutionEnv ec = CoevolutionaryRuleExtractor.constructEnvironmentForWEKAInstances(adapter);
        evoGUI(classifier);
        return classifier;
    }

    private static CoevolutionaryRuleExtractor coev(TrainAndTestInstances data) throws HeadlessException {
        CoevolutionaryRuleExtractor classifier = new CoevolutionaryRuleExtractor();
        classifier.setGenerations(generations);
        DataAdapter adapter = new DataAdapter(data.train());
        ExecutionEnv ec = CoevolutionaryRuleExtractor.constructEnvironmentForWEKAInstances(adapter);
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
