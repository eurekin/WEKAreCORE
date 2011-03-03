package core.tests;

import core.adapters.StockSets;
import core.adapters.TrainAndTestInstances;
import java.io.FileWriter;
import java.io.IOException;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.CoevolutionaryRuleExtractor;

/**
 *
 * @author gmatoga
 */
public class CopopulationsSize {

    public static void main(String[] args) throws InterruptedException, ExecutionException, IOException {
        int threads = Runtime.getRuntime().availableProcessors();
        ScheduledExecutorService exs = Executors.newScheduledThreadPool(threads);
        CompletionService<String> pool = new ExecutorCompletionService<String>(exs);

        int howMany = 0;
        int reps = 15;
        for (TrainAndTestInstances set : new TrainAndTestInstances[] {StockSets.glass()}) {
            for (int rules : new int[]{15, 20, 50, 100, 200, 500, 1000}) {
                for (int rulesets : new int[]{15, 20, 50, 100, 200, 500, 1000}) {
                    for (int i = 0; i < reps; i++) {
                        pool.submit(new TestTask(set, rules, rulesets));
                        howMany++;
                    }
                }
            }
        }
        exs.shutdown();
        String get;
        FileWriter out = new FileWriter("copop-size-results1.csv");
        for (int i = 0; i < howMany; i++) {
            get = pool.take().get();
            System.out.println(get);
            out.write(get.replace(".", ",") + "\n");
            out.flush();
        }
        out.close();
    }

    private static class TestTask implements Callable<String> {

        TrainAndTestInstances set;
        int rules;
        int rulesets;

        public TestTask(TrainAndTestInstances set, int rules, int rulesets) {
            this.set = set;
            this.rules = rules;
            this.rulesets = rulesets;
        }

        public String call() throws Exception {
            Evaluation eval = new Evaluation(set.train());
            CoevolutionaryRuleExtractor classifier = new CoevolutionaryRuleExtractor();
            classifier.setGenerations(200);
            classifier.setTokenCompetitionEnabled(false);
            classifier.setSelection(0);
            classifier.setEliteSelectionSize(0);
            classifier.setRuleSetPopulationSize(rulesets);
            classifier.setRulePopulationSize(rules);

            classifier.buildClassifier(set.train());
            eval.evaluateModel(classifier, set.test());
            return set.train().relationName() + ";" + rules + ";" + rulesets + ";" + eval.pctCorrect();
        }
    }
}
