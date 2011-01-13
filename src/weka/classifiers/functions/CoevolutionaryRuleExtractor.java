package weka.classifiers.functions;

import core.adapters.DataAdapter;
import core.copop.CoPopulations;
import core.copop.RuleSet;
import core.ga.DefaultEvaluator;
import core.ga.GrayBinaryDecoderPlusONE;
import core.ga.RuleDecoderSubractingOneFromClass;
import core.ga.ops.ec.ExecutionEnv;
import core.ga.ops.ec.FitnessEval;
import core.ga.ops.ec.FitnessEvaluatorFactory;
import core.vis.CoordCalc;
import core.vis.RuleASCIIPlotter;
import java.beans.BeanInfo;
import java.beans.Introspector;
import java.beans.PropertyDescriptor;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import java.util.Vector;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.RandomizableClassifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;

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
public class CoevolutionaryRuleExtractor extends RandomizableClassifier {

    private RuleSet best;
    private String bestString;

    @Override
    public void buildClassifier(Instances data) throws Exception {


        // mock MONK dataset
        FitnessEval fit = FitnessEvaluatorFactory.EVAL_ACCURACY;

        // here is the junction
        // when DataAdapter is set as a bundle then weka's instances are used
        DataAdapter adapter = new DataAdapter(data);
        final DefaultEvaluator eval = new DefaultEvaluator();
        final GrayBinaryDecoderPlusONE bdec = new GrayBinaryDecoderPlusONE();
        final RuleDecoderSubractingOneFromClass dec =
                new RuleDecoderSubractingOneFromClass(adapter.getBundle().getSignature(), bdec);
        ExecutionEnv ec = new ExecutionEnv(1000, new Random(m_Seed), eval, adapter.getBundle(), dec, fit);
        final RuleASCIIPlotter plotter = ec.getBundle().getPlotter();

        ///// Try to visualize initial problem
        ArrayList comb;
        Enumeration instances = data.enumerateInstances();
        ArrayList<Instance> list = Collections.list(instances);
        CoordCalc c = new CoordCalc(ec.signature());
        String[][] datavis = RuleASCIIPlotter.initEmptyDataVis(c);
        for (Instance instance : list) {
            comb = new ArrayList();

            for (int i = 0; i < instance.numAttributes() - 1; i++) {
                comb.add((int) instance.value(i));
            }
            datavis[c.getY(comb)][c.getX(comb)] =
                    new Integer((int) instance.value(instance.numAttributes() - 1)).toString();
        }
        RuleASCIIPlotter.simpleBinaryPlot(datavis);
        System.out.println("About to plot training data:");
        plotter.plotPlots(datavis);



        // Set coevolution params
        long seed = System.currentTimeMillis();
        ec.rand().setSeed(seed);

        ec.setMt(ruleMutationProbability);
        ec.setRsmp(ruleSetMutationProbability);
        ec.setMaxRuleSetLength(maxRulesCount);
        ec.setRulePopSize(rulePopulationSize);
        ec.setTokenCompetitionEnabled(tokenCompetitionEnabled);

        spitOutOptions();


        co = new CoPopulations(ruleSetPopulationSize, ec);
        co.setDebug(true);

        // evolution
        System.out.println("Starting coevolution");
        int t = generations;
        while (t-- > 0)
            co.evolve();
        System.out.println("Coevolution finished");

        // final report
        best = co.getBest().getRS();
        System.out.println("The stats are "
                           + co.ruleSets().getBest().getCm().getWeighted());
        System.out.println("The final classifier:");
        System.out.println("Visualization: ");
        plotter.detailedPlots(best);
        bestString =
                ec.getBundle().getPrinter().print(best);
    }
    transient CoPopulations co;

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        ArrayList<Integer> list = new ArrayList<Integer>();
        for (int i = 0; i < instance.numValues() - 1; i++) {
            list.add((int) instance.value(i));
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
        result.enable(Capability.NOMINAL_CLASS);
        return result;
    }

    /**
     * Main method for testing this class.
     */
    public static void main(String[] argv) {
        int M = 1;
        URL train = CoevolutionaryRuleExtractor.class.getResource(
                "/monks/monks-" + M + ".train.arff");
        URL test = CoevolutionaryRuleExtractor.class.getResource(
                "/monks/monks-" + M + ".test.arff");
        // Instances.main(new String[] {resource.getPath()});
        String[] args = new String[]{
            "-i", // per class statistics
            "-t", train.getPath(), // train set
            "-T", test.getPath() // test set (will go with 10-fold CV if not set)
        };
        System.out.println("args = " + Arrays.deepToString(args));
        runClassifier(new CoevolutionaryRuleExtractor(), args);
    }

    @Override
    public String toString() {
        if (bestString != null && !bestString.isEmpty())
            return bestString;
        else
            return "reCORE";
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

        if (isTokenCompetitionEnabled())
            ol.add("-T");


        return ol.toArray(new String[]{});
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        if (ok(Utils.getOption('G', options)))
            setGenerations(Integer.parseInt(Utils.getOption('G', options)));
        if (ok(Utils.getOption("-MM", options)))
            setRuleMutationProbability(Double.parseDouble(Utils.getOption("-MM", options)));
        if (ok(Utils.getOption("-MP", options)))
            setRuleSetMutationProbability(Double.parseDouble(Utils.getOption("-MP", options)));
        if (ok(Utils.getOption('R', options)))
            setMaxRulesCount(Integer.parseInt(Utils.getOption('R', options)));
        if (ok(Utils.getOption("-CM", options)))
            setRulePopulationSize(Integer.parseInt(Utils.getOption("-CM", options)));
        if (ok(Utils.getOption("-CP", options)))
            setRuleSetPopulationSize(Integer.parseInt(Utils.getOption("-CP", options)));
        if (Utils.getOptionPos("T", options) != -1)
            setTokenCompetitionEnabled(Utils.getFlag("T", options));

        super.setOptions(options);
    }
    // OPTIONS
    private int generations = 100;
    double ruleMutationProbability = 1.0E-4;
    double ruleSetMutationProbability = 0.2;
    int maxRulesCount = 9;
    int rulePopulationSize = 1000;
    int ruleSetPopulationSize = 1000;
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
        try {
            BeanInfo info = Introspector.getBeanInfo(this.getClass());
            System.out.println("Using following properties: ");
            for (PropertyDescriptor pd : info.getPropertyDescriptors()) {
                System.out.print(padRight(pd.getName(), 30));
                System.out.print("\t");
                Object result = pd.getReadMethod().invoke(this);
                if (result instanceof Object[])
                    System.out.println(Arrays.deepToString((Object[]) result));
                else
                    System.out.println(result);
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
    private boolean ok(String option) {
        return option != null && option.length() != 0 && !option.isEmpty();
    }
}
