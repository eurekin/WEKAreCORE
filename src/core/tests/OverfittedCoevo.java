package core.tests;

import core.adapters.StockSets;
import core.adapters.TrainAndTestInstances;
import core.tests.math.Ranges;
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
public class OverfittedCoevo {

    public static void main(String[] args) throws InterruptedException, ExecutionException, IOException {
        int threads = Runtime.getRuntime().availableProcessors();
        ScheduledExecutorService exs = Executors.newScheduledThreadPool(threads);
        CompletionService<String> pool = new ExecutorCompletionService<String>(exs);

        int howMany = 0;
        int reps = 5;
        for (int gens = 50; gens < 15000; gens+=50) {
                for (int i = 0; i < reps; i++) {
                    pool.submit(new TestTask(gens));
                    howMany++;
                }
        }
        exs.shutdown();
        String get;
        FileWriter out = new FileWriter("ovr-results1.csv");
        for (int i = 0; i < howMany; i++) {
            get = pool.take().get();
            System.out.println(get);
            out.write(get.replace(".", ",") + "\n");
            out.flush();
        }
        out.close();
    }
    private static final TrainAndTestInstances set = StockSets.monks1();

    private static class TestTask implements Callable<String> {

        int gens;

        public TestTask(int gens) {
            this.gens = gens;
        }

        public String call() throws Exception {
            CoevolutionaryRuleExtractor classifier = new CoevolutionaryRuleExtractor();
            classifier.setGenerations(gens);
            classifier.setTokenCompetitionEnabled(false);

            classifier.buildClassifier(set.train());

            Evaluation evalOnTrain = new Evaluation(set.train());
            evalOnTrain.evaluateModel(classifier, set.train());

            Evaluation evalOnTest = new Evaluation(set.test());
            evalOnTest.evaluateModel(classifier, set.test());
            
            return set.train().relationName() + ";" + gens + ";" + evalOnTrain.pctCorrect() + ";" + evalOnTest.pctCorrect();

        }
    }
}
