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
public class TokenCompetitionCoevo {

    public static void main(String[] args) throws InterruptedException, ExecutionException, IOException {
        int threads = Runtime.getRuntime().availableProcessors();
        ScheduledExecutorService exs = Executors.newScheduledThreadPool(threads);
        CompletionService<String> pool = new ExecutorCompletionService<String>(exs);

        int howMany = 0;
        int reps = 30;
        for (TrainAndTestInstances set : StockSets.SETS) {
                for (int i = 0; i < reps; i++) {
                    pool.submit(new TestTask(set, true));
                    pool.submit(new TestTask(set, false));
                    howMany++;
                    howMany++;
                }
        }
        exs.shutdown();
        String get;
        FileWriter out = new FileWriter("tok-results.csv");
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
        boolean tokon;

        public TestTask(TrainAndTestInstances set, boolean tokon) {
            this.set = set;
            this.tokon = tokon;
        }

        public String call() throws Exception {
            Evaluation eval = new Evaluation(set.train());
            CoevolutionaryRuleExtractor classifier = new CoevolutionaryRuleExtractor();
            classifier.setGenerations(200);
            classifier.setTokenCompetitionEnabled(tokon);
            classifier.buildClassifier(set.train());
            eval.evaluateModel(classifier, set.test());
            return set.train().relationName() + ";" + tokon + ";" + eval.pctCorrect();
        }
    }
}
