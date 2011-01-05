package weka.classifiers.functions;

import core.ExecutionContextFactory;
import core.copop.CoPopulations;
import core.copop.RuleSet;
import core.ga.ops.ec.ExecutionContext;
import core.ga.ops.ec.FitnessEval;
import core.ga.ops.ec.FitnessEvaluatorFactory;
import core.vis.CoordCalc;
import core.vis.RuleASCIIPlotter;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import weka.classifiers.RandomizableClassifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author gmatoga
 */
public class CoevolutionaryRuleExtractor extends RandomizableClassifier {

    private RuleSet best;
    ArrayList expected = new ArrayList();

    @Override
    public void buildClassifier(Instances data) throws Exception {
        Enumeration instances = data.enumerateInstances();
        ArrayList<Instance> list = Collections.list(instances);

        ExecutionContext ec;
        FitnessEval fit = FitnessEvaluatorFactory.EVAL_FMEASURE;
        ec = ExecutionContextFactory.MONK(1, false, 1000, fit);
        ///// Try to visualize
        ArrayList comb;
        CoordCalc c = new CoordCalc(ec.signature());
        String[][] datavis = RuleASCIIPlotter.initEmptyDataVis(c);
        for (Instance instance : list) {
            comb = new ArrayList();

            for (int i = 0; i < 6; i++) {
                comb.add((int) instance.value(i)+1);
            }
            datavis[c.getY(comb)][c.getX(comb)] = instance.value(6)!=0.0d ? " " : "#";
        }
        RuleASCIIPlotter.simplePlot(datavis);

//        calc.getX(expected);
//        calc.getY(expected);
//        RuleASCIIPlotter.simplePlot(
//        RuleASCIIPlotter.getPlot(ec.signature(), new Plotable() {
//
//            public String call(List<Integer> comb) {
//                return m3(comb);
//            }
//        }));

        long seed = System.currentTimeMillis();
        ec.rand().setSeed(seed);
        ec.setMt(0.02);
        ec.setRsmp(0.001);
        ec.setMaxRuleSetLength(5);
        co = new CoPopulations(1000, ec);
        int t = 170;
        while (t-- > 0)
            co.evolve();
        best = co.getBest().getRS();
        System.out.println("Visualization: ");
        String[][] plot = ec.getBundle().getPlotter().plotRuleSet(best);
        RuleASCIIPlotter.simplePlot(plot);
        System.out.println("The stats are " + co.ruleSets().getBest().getCm().getWeighted());
    }
    CoPopulations co;

    @Override
    public Enumeration listOptions() {
        return super.listOptions();
    }
    int id = 0;

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        ArrayList<Integer> list = new ArrayList<Integer>();
        for (int i = 0; i < 6; i++) {
            list.add((int) instance.value(i)+1);
        }
        int result = best.apply(list);
        return 1 - result;
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();   // returns the object from weka.classifiers.Classifier
        result.disableAll();
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NOMINAL_CLASS);
        return result;
    }

    /**
     * Main method for testing this class.
     */
    public static void main(String[] argv) {
        URL resource = CoevolutionaryRuleExtractor.class.getResource("/monks/monks-1.test.arff");
        // Instances.main(new String[] {resource.getPath()});
        String[] args = new String[]{"-i", "-t", resource.getPath(), "-T", resource.getPath()};
        runClassifier(new CoevolutionaryRuleExtractor(), args);
    }
}
