package weka.bin;

import java.util.Locale;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import weka.core.Instances;
import weka.core.Utils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.CoevolutionaryRuleExtractor;
import weka.classifiers.functions.EvolutionaryRuleExtractor;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.RBFNetwork;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48graft;
import core.adapters.TrainAndTestInstances;

/**
 * Performs multiple runs of cross-validation.
 *
 * Command-line parameters:
 * <ul>
 *    <li>-t filename - the dataset to use</li>
 *    <li>-x int - the number of folds to use</li>
 *    <li>-r int - the number of runs to perform</li>
 *    <li>-c int - the class index, "first" and "last" are accepted as well;
 *    "last" is used by default</li>
 *    <li>-W classifier - classname and options, enclosed by double quotes;
 *    the classifier to cross-validate</li>
 * </ul>
 *
 * Example command-line:
 * <pre>
 * java CrossValidationMultipleRuns -t labor.arff -c last -x 10 -r 10 -W
 * "weka.classifiers.trees.J48 -C 0.25"
 * </pre>
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
public class CrossValidationMultipleRuns {

    public static void main(String[] args) throws Exception {
        int threadno = Runtime.getRuntime().availableProcessors();
        ExecutorService es = Executors.newFixedThreadPool(threadno);
        ResultCollector collector = new ResultCollector();

        for (String set : "diabetes,iris,weather,glass".split(",")) {
            for (int i = 0; i < threadno; i++) {
                es.submit(new Task(i, threadno, collector, set));
            }
        }
        es.shutdown();
    }

    /**
     * Performs the cross-validation. See Javadoc of class for information
     * on command-line parameters.
     *
     * @param args        the command-line parameters
     * @throws Excecption if something goes wrong
     */
    public static void run2(int no, int total, ResultCollector collector,
            String set) throws Exception {
        // loads data and set class index
        TrainAndTestInstances tati = new TrainAndTestInstances(set + ".train", set + ".test");
        Instances data = tati.train();
        Locale.setDefault(Locale.ENGLISH);
        // classifier
        CoevolutionaryRuleExtractor core = new CoevolutionaryRuleExtractor();
        core.setGenerations(1000);
        core.setRuleMutationProbability(0.01);
        core.setRuleSetMutationProbability(0.01);
        core.setMaxRulesCount(9);
        EvolutionaryRuleExtractor evol = new EvolutionaryRuleExtractor();
        evol.setGenerations(1000);
        evol.setRuleMutationProbability(0.01);
        evol.setRuleSetMutationProbability(0.01);
        evol.setMaxRulesCount(9);

        Classifier[] classifiers = new Classifier[]{
            core,
            evol,
            new J48graft(),
            new NaiveBayes(),
            new BayesNet(),
            new SMO(),
            new MultilayerPerceptron(),
            new RBFNetwork()};

        // other options
        int runs = 5;
        int folds = 10;
        // perform cross-validation
        for (int i = 0; i < runs; i++) {
            // randomize data

                for (Classifier classifier : classifiers) {
                    Evaluation eval = new Evaluation(data);
                    Instances train = tati.train();
                    Instances test = tati.test();


                    // build and evaluate classifier
                    Classifier clsCopy = Classifier.makeCopy(classifier);
                    clsCopy.buildClassifier(train);
                    eval.evaluateModel(clsCopy, test);
                    String format = String.format("%s,%s,%d,%.8f,%d",
                            classifier.getClass().getCanonicalName().substring(
                            classifier.getClass().getCanonicalName().lastIndexOf('.') + 1),
                            set,
                            i, eval.pctCorrect(), ((int) eval.correct()));
                    collector.collect(format);
            }
//            outputEvaluationWEKAstyle(i, cls, data, folds, seed, eval);
        }
    }

    public static class Task extends Thread {

        final int no, total;
        private final ResultCollector collector;
        String set;

        public Task(int no, int total, ResultCollector collector, String set) {
            this.no = no;
            this.total = total;
            this.collector = collector;
            this.set = set;
        }

        @Override
        public void run() {
            try {
                run2(no, total, collector, set);
            } catch (Exception ex) {
                Logger.getLogger(
                        CrossValidationMultipleRuns.class.getName()).log(
                        Level.SEVERE, null, ex);
            }
        }
    }

    private static void outputEvaluationWEKAstyle(int i, Classifier cls,
            Instances data, int folds, int seed, Evaluation eval) {
        // output evaluation
        System.out.println();
        System.out.println("=== Setup run " + (i + 1) + " ===");
        System.out.println("Classifier: " + cls.getClass().getName()
                + " " + Utils.joinOptions(cls.getOptions()));
        System.out.println("Dataset: " + data.relationName());
        System.out.println("Folds: " + folds);
        System.out.println("Seed: " + seed);
        System.out.println();
        System.out.println(eval.toSummaryString("=== " + folds
                + "-fold Cross-validation run " + (i + 1) + "===", false));
    }

    public static boolean notMyTurn(int seq, int no, int total) {
        return !(seq % total == no);
    }

    public static class ResultCollector {

        public ResultCollector() {
        }

        public synchronized void collect(Object result) {
            System.out.println(result);
            System.out.flush();
        }
    }
}
