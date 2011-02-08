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
public class SelectionTypesTest {

    public static void main(String[] args) throws InterruptedException, ExecutionException, IOException {
        int granularity = 20;
        int startExp = 0;
        int endExp = -4;

        int threads = Runtime.getRuntime().availableProcessors();
        ScheduledExecutorService exs = Executors.newScheduledThreadPool(threads);
        CompletionService<String> pool = new ExecutorCompletionService<String>(exs);

        int howMany = 0;
        int reps = 15;
        for (Double d : Ranges.logrange(startExp, endExp, granularity)) {
            for (Double d2 : Ranges.logrange(startExp, endExp, granularity)) {
                for (int i = 0; i < reps; i++) {
                    pool.submit(new TestTask(d, d2));
                    howMany++;
                }
            }
        }
        exs.shutdown();
        String get;
        FileWriter out = new FileWriter("grid-results.csv");
        for (int i = 0; i < howMany; i++) {
            get = pool.take().get();
            System.out.println(get);
            out.write(get.replace(".", ",") + "\n");
            out.flush();
        }
        out.close();
    }

    private static class TestTask implements Callable<String> {

        double d, d2;
        public static final ThreadLocal<TrainAndTestInstances> data =
                new ThreadLocal<TrainAndTestInstances>() {

                    @Override
                    protected TrainAndTestInstances initialValue() {
                        return StockSets.iris();
                    }
                };

        public TestTask(double d, double d2) {
            this.d = d;
            this.d2 = d2;
        }

        public String call() throws Exception {
            Evaluation eval = new Evaluation(data.get().train());
            CoevolutionaryRuleExtractor classifier = new CoevolutionaryRuleExtractor();
            classifier.setRuleMutationProbability(d);
            classifier.setRuleSetMutationProbability(d2);
            classifier.setGenerations(200);
            classifier.setTokenCompetitionEnabled(false);
            classifier.buildClassifier(data.get().train());
            eval.evaluateModel(classifier, data.get().test());
            return data.get().train().relationName() + ";" + d + ";" + d2 + ";" + eval.pctCorrect();
        }
    }
}
