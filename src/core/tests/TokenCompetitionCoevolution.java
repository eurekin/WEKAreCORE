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
import weka.classifiers.functions.EvolutionaryRuleExtractor;

/**
 *
 * @author gmatoga
 */
public class TokenCompetitionCoevolution {

    public static void main(String[] args) throws InterruptedException, ExecutionException, IOException {
        int granularity = 200;
        int startExp = 0;
        int endExp = -6;

        int threads = Runtime.getRuntime().availableProcessors();
        ScheduledExecutorService exs = Executors.newScheduledThreadPool(threads);
        CompletionService<String> pool = new ExecutorCompletionService<String>(exs);

        int howMany = 0;
        int reps = 5;
        for (TrainAndTestInstances set : StockSets.SETS) {
            for (Double d : Ranges.logrange(startExp, endExp, granularity)) {
                for (int i = 0; i < reps; i++) {
                    pool.submit(new TestTask(d, set));
                    howMany++;
                }
            }
        }
        exs.shutdown();
        String get;
        FileWriter out = new FileWriter("result.csv");
        for (int i = 0; i < howMany; i++) {
            get = pool.take().get();
            System.out.println(get);
            out.write(get.replace(".", ",") + "\n");
            out.flush();
        }
        out.close();
    }

    private static class TestTask implements Callable<String> {

        double d;
        public final TrainAndTestInstances data;

        public TestTask(double d, TrainAndTestInstances data) {
            this.d = d;
            this.data = data;
        }

        public String call() throws Exception {
            Evaluation eval = new Evaluation(data.train());
            EvolutionaryRuleExtractor classifier = new EvolutionaryRuleExtractor();
            classifier.setRuleMutationProbability(d);
            classifier.setRuleSetMutationProbability(d);
            classifier.setGenerations(200);
            classifier.setTokenCompetitionEnabled(false);
            classifier.buildClassifier(data.train());
            eval.evaluateModel(classifier, data.test());
            return data.train().relationName() + ";" + d + ";" + eval.pctCorrect();
        }
    }
}
