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
public class MutationGridCoevolution {

    public static void main(String[] args) throws InterruptedException, ExecutionException, IOException {
        int granularity = 20;
        int startExp = 0;
        int endExp = -4;

        int threads = Runtime.getRuntime().availableProcessors();
        ScheduledExecutorService exs = Executors.newScheduledThreadPool(threads);
        CompletionService<String> pool = new ExecutorCompletionService<String>(exs);

        int howMany = 0;
        int reps = 15;
        for (TrainAndTestInstances set : StockSets.SETS) {
            for (int seltype : new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
                for (int i = 0; i < reps; i++) {
                    pool.submit(new TestTask(set, seltype));
                    howMany++;
                }
            }
        }
        exs.shutdown();
        String get;
        FileWriter out = new FileWriter("mut-results.csv");
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
        int seltype;

        public TestTask(TrainAndTestInstances set, int seltype) {
            this.set = set;
            this.seltype = seltype;
        }

        public String call() throws Exception {
            Evaluation eval = new Evaluation(set.train());
            CoevolutionaryRuleExtractor classifier = new CoevolutionaryRuleExtractor();
            classifier.setGenerations(200);
            classifier.setTokenCompetitionEnabled(false);
            classifier.setSelection(seltype);
            classifier.buildClassifier(set.train());
            eval.evaluateModel(classifier, set.test());
            return set.train().relationName() + ";" + seltype + ";" + eval.pctCorrect();
        }
    }
}
