package weka.classifiers.functions;

import core.ga.RulePrinter;
import core.io.repr.col.Domain;
import core.io.repr.col.FloatDomain;
import core.adapters.DataAdapter;
import core.copop.RuleSet;
import core.evo.EvolutionPopulation;
import core.ga.ops.ec.ExecutionEnv;
import core.vis.RuleASCIIPlotter;
import java.beans.BeanInfo;
import java.beans.Introspector;
import java.beans.PropertyDescriptor;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.reflect.Method;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.Vector;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.RandomizableClassifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import static weka.classifiers.functions.CoevolutionaryRuleExtractor.*;

/**
 *
 * Success is not final, failure is not fatal: it is the courage to continue
 * that counts.
 *                                                     --- Winston Churchill
 *
 *
 * Opcje:
 * pr mut reg
 * pr mut zbioru reg
 * liczebnosc pop reg
 * liczebnosc pop zb reg
 * ilosc pokoleń
 * token competition
 * rozmiar selekcji elitarnej
 *
 * @author gmatoga
 */
public class EvolutionaryRuleExtractor extends RandomizableClassifier {

    private static double evalOnMonk(int M, Classifier classifier)
            throws FileNotFoundException, Exception, IOException {
        URL train = CoevolutionaryRuleExtractor.class.getResource(
                "/monks/monks-" + M + ".train.arff");
        URL test = CoevolutionaryRuleExtractor.class.getResource(
                "/monks/monks-" + M + ".test.arff");
        // Instances.main(new String[] {resource.getPath()});
        String[] args = new String[]{ //            "-i",
        //            "-t", train.getPath()
        //            "-T", test.getPath()
        };
        //System.out.println("args = " + Arrays.deepToString(args));
        //runClassifier(new CoevolutionaryRuleExtractor(), args);
        PrintStream a = System.out;
        File createTempFile = File.createTempFile("ignoreme", "tmp");
        System.setOut(new PrintStream(createTempFile));
        Instances trainData = DataSource.read(train.getPath());
        Instances testData = DataSource.read(test.getPath());
        trainData.setClassIndex(trainData.numAttributes() - 1);
        testData.setClassIndex(testData.numAttributes() - 1);
        classifier.setOptions(args);
        classifier.buildClassifier(trainData);
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(classifier, testData);
        System.setOut(a);
        return eval.pctCorrect();
    }
    private RuleSet best;
    private String bestString;
    transient EvolutionPopulation co;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        DataAdapter adapter = new DataAdapter(data);
        ExecutionEnv ec = constructEnvironmentForWEKAInstances(adapter);
        saveDomainTypesForFutureEvaluationWhichUsesSerialization(ec);

        final RuleASCIIPlotter plotter = ec.getBundle().getPlotter();

        if (plotter != null) {
            CoevolutionaryRuleExtractor.visualizeData(data, ec, plotter);
        }

        // Set Evolution params
        long seed = System.currentTimeMillis();
        ec.rand().setSeed(seed);

        ec.setMt(ruleMutationProbability);
        ec.setRsmp(ruleSetMutationProbability);
        ec.setMaxRuleSetLength(maxRulesCount);
        ec.setRulePopSize(rulePopulationSize);
        ec.setRuleSetCount(ruleSetPopulationSize);
        ec.setTokenCompetitionEnabled(tokenCompetitionEnabled);
        ec.setTokenCompetitionWeight(1.0);
        ec.setEliteSelectionSize(1);

        spitOutOptions();


        co = new EvolutionPopulation(ec);

        // evolution
        if (getDebug())
            System.out.println("Starting Evolution");
        int t = generations;
        while (t-- > 0) {
            co.evolve();
            callBack();
            if (co.getBest().fitness() == 1.0d) {
                break;
            }
        }
        RulePrinter printer = ec.getBundle().getPrinter();
        best = co.getBest().getRS();
        if (printer != null) {
            bestString = printer.print(best);
        }

        if (!getDebug())
            return;
        System.out.println("Evolution finished");

        // final report
        System.out.println("The stats are " + co.getBest().getCm().getWeighted());
        if (plotter != null) {
            System.out.println("The final classifier:");
            System.out.println("Visualization: ");
            plotter.detailedPlots(best);
        }

    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        ArrayList<Object> list = new ArrayList<Object>();
        for (int i = 0; i < isNumericAttribute.size(); i++) {
            Boolean isNumeric = isNumericAttribute.get(i);
            if (isNumeric) {
                list.add((float) instance.value(i));
            } else {
                list.add((int) instance.value(i));
            }
        }
        int result = best.apply(list);
        return result;
    }

    @Override
    public Capabilities getCapabilities() {
        // returns the object from weka.classifiers.Classifier
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.NOMINAL_CLASS);
        return result;
    }

    /**
     * Main method for testing this class.
     */
    public static void main(String[] argv) throws Exception {
        int M = 3;
        for (int i = 0; i < 50; i++) { // tyyle powtórzeń
            for (int j = 1; j < 4; j++) { // dla każdego z MONKów
                Classifier[] classifiers = new Classifier[]{
                    new EvolutionaryRuleExtractor(),
                    new CoevolutionaryRuleExtractor(), //                    new J48graft(),
                //                    new ZeroR(),
                //                    new OneR(),
                //                    new MultilayerPerceptron(),
                //                    new BayesNet(),
                //                    new NaiveBayes(),
                //                    new RBFNetwork(),
                //                    new SMO()
                };
                for (Classifier classifier : classifiers) {

                    double evalOnMonk = evalOnMonk(j, classifier);
                    System.out.println(classifier.getClass().getName()
                                       + ", " + i + ", " + j + ", " + evalOnMonk);
                }
            }
        }
    }

    @Override
    public String toString() {
        if (bestString != null && !bestString.isEmpty()) {
            return bestString;
        } else {
            return "reCORE";
        }
    }

    @Override
    public Enumeration listOptions() {
        Vector newVector = new Vector(10);

        newVector.addElement(new Option(
                "Number of generations to evolve",
                "G", 1, "-G <generations>"));

        newVector.addElement(new Option(
                "Turn on or off token competition.",
                "T", 0, "-T"));

        newVector.addElement(new Option(
                "Rule mutation probability.",
                "MM", 0, "-MM <double>"));

        newVector.addElement(new Option(
                "Rule set mutation probability.",
                "MP", 0, "-MP <double>"));

        newVector.addElement(new Option(
                "Maximum rule count.",
                "R", 1, "-R <rules>"));

        newVector.addElement(new Option(
                "Rule population size.",
                "CM", 0, "-CM <count>"));

        newVector.addElement(new Option(
                "Rule set population size.",
                "CP", 0, "-CP <count>"));

        Enumeration enu = super.listOptions();
        while (enu.hasMoreElements()) {
            newVector.addElement(enu.nextElement());
        }
        return newVector.elements();
    }

    @Override
    public String[] getOptions() {
        List<String> ol = new ArrayList<String>(Arrays.asList(super.getOptions()));

        ol.add("-G");
        ol.add(String.valueOf(getGenerations()));

        ol.add("-MM");
        ol.add(String.valueOf(getRuleMutationProbability()));

        ol.add("-MP");
        ol.add(String.valueOf(getRuleSetMutationProbability()));

        ol.add("-R");
        ol.add(String.valueOf(getMaxRulesCount()));

        ol.add("-CM");
        ol.add(String.valueOf(getRulePopulationSize()));

        ol.add("-CP");
        ol.add(String.valueOf(getRuleSetPopulationSize()));

        if (isTokenCompetitionEnabled()) {
            ol.add("-T");
        }


        return ol.toArray(new String[]{});
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        if (ok(Utils.getOption('G', options))) {
            setGenerations(Integer.parseInt(Utils.getOption('G', options)));
        }
        if (ok(Utils.getOption("-MM", options))) {
            setRuleMutationProbability(Double.parseDouble(Utils.getOption("-MM", options)));
        }
        if (ok(Utils.getOption("-MP", options))) {
            setRuleSetMutationProbability(Double.parseDouble(Utils.getOption("-MP", options)));
        }
        if (ok(Utils.getOption('R', options))) {
            setMaxRulesCount(Integer.parseInt(Utils.getOption('R', options)));
        }
        if (ok(Utils.getOption("-CM", options))) {
            setRulePopulationSize(Integer.parseInt(Utils.getOption("-CM", options)));
        }
        if (ok(Utils.getOption("-CP", options))) {
            setRuleSetPopulationSize(Integer.parseInt(Utils.getOption("-CP", options)));
        }
        if (Utils.getOptionPos("T", options) != -1) {
            setTokenCompetitionEnabled(Utils.getFlag("T", options));
        }

        super.setOptions(options);
    }
    // OPTIONS
    private int generations = 1500;
    double ruleMutationProbability = 0.02;
    double ruleSetMutationProbability = 0.15;
    int maxRulesCount = 10;
    int rulePopulationSize = 200;
    int ruleSetPopulationSize = 200;
    private boolean tokenCompetitionEnabled = true;

    public boolean isTokenCompetitionEnabled() {
        return tokenCompetitionEnabled;
    }

    public void setTokenCompetitionEnabled(boolean tokenCompetitionEnabled) {
        this.tokenCompetitionEnabled = tokenCompetitionEnabled;
    }

    public int getGenerations() {
        return generations;
    }

    public void setGenerations(int generations) {
        this.generations = generations;
    }

    public int getMaxRulesCount() {
        return maxRulesCount;
    }

    public void setMaxRulesCount(int maxRulesCount) {
        this.maxRulesCount = maxRulesCount;
    }

    public double getRuleMutationProbability() {
        return ruleMutationProbability;
    }

    public void setRuleMutationProbability(double ruleMutationProbability) {
        this.ruleMutationProbability = ruleMutationProbability;
    }

    public int getRulePopulationSize() {
        return rulePopulationSize;
    }

    public void setRulePopulationSize(int rulePopulationSize) {
        this.rulePopulationSize = rulePopulationSize;
    }

    public double getRuleSetMutationProbability() {
        return ruleSetMutationProbability;
    }

    public void setRuleSetMutationProbability(double ruleSetMutationProbability) {
        this.ruleSetMutationProbability = ruleSetMutationProbability;
    }

    public int getRuleSetPopulationSize() {
        return ruleSetPopulationSize;
    }

    public void setRuleSetPopulationSize(int ruleSetPopulationSize) {
        this.ruleSetPopulationSize = ruleSetPopulationSize;
    }

    private void spitOutOptions() {
        if (!getDebug())
            return;

        try {
            BeanInfo info = Introspector.getBeanInfo(this.getClass());
            System.out.println("Using following properties: ");
            for (PropertyDescriptor pd : info.getPropertyDescriptors()) {
                Method readMethod = pd.getReadMethod();
                if (readMethod == null) {
                    continue;
                }
                System.out.print(padRight(pd.getName(), 30));
                System.out.print("\t");
                Object result = readMethod.invoke(this);
                if (result instanceof Object[]) {
                    System.out.println(Arrays.deepToString((Object[]) result));
                } else {
                    System.out.println(result);
                }
            }

        } catch (Exception ex) {
            Logger.getLogger(CoevolutionaryRuleExtractor.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static String padRight(String s, int n) {
        return String.format("%1$-" + n + "s", s);
    }

    public static String padLeft(String s, int n) {
        return String.format("%1$#" + n + "s", s);
    }
    // For debug only;
    transient private EvolutionCallback callback;

    private boolean ok(String option) {
        return option != null && option.length() != 0 && !option.isEmpty();
    }

    public void setCallback(EvolutionCallback coevolutionCallback) {
        this.callback = coevolutionCallback;
    }
    List<Boolean> isNumericAttribute;

    private void callBack() {
        if (callback != null) {
            callback.coevolutionCallback(co);
        }
    }

    private void saveDomainTypesForFutureEvaluationWhichUsesSerialization(ExecutionEnv ec) {
        isNumericAttribute = new ArrayList<Boolean>();
        for (Domain domain : ec.signature().getAttrDomain()) {
            isNumericAttribute.add(domain instanceof FloatDomain);
        }
    }
}
