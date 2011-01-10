package weka.classifiers.functions;

import core.ExecutionContextFactory;
import core.adapters.DataAdapter;
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


        // mock MONK dataset
        ExecutionContext ec;
        FitnessEval fit = FitnessEvaluatorFactory.EVAL_ACCURACY;
        ec = ExecutionContextFactory.MONK(1, false, 1000, fit);

        // here is the junction
        // when DataAdapter is set as a bundle then weka's instances are used
        DataAdapter adapter = new DataAdapter(data);
        ec.setBundle(adapter.getBundle());
        final RuleASCIIPlotter plotter = ec.getBundle().getPlotter();

        ///// Try to visualize initial problem
        ArrayList comb;
        Enumeration instances = data.enumerateInstances();
        ArrayList<Instance> list = Collections.list(instances);
        CoordCalc c = new CoordCalc(ec.signature());
        String[][] datavis = RuleASCIIPlotter.initEmptyDataVis(c);
        for (Instance instance : list) {
            comb = new ArrayList();

            for (int i = 0; i < 6; i++) {
                comb.add((int) instance.value(i));
            }
            datavis[c.getY(comb)][c.getX(comb)] =
                    new Integer((int) instance.value(6)).toString();
        }
//        RuleASCIIPlotter.simpleBinaryPlot(datavis);
        plotter.plotPlots(datavis);



        // Set coevolution params
        long seed = System.currentTimeMillis();
        ec.rand().setSeed(seed);
        ec.setMt(0.0001);
        ec.setRsmp(0.2);
        ec.setMaxRuleSetLength(10);
        co = new CoPopulations(1000, ec);
        co.setDebug(true);
        int t = 50;

        // evolution
        System.out.println("Starting coevolution");
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
        URL resource = CoevolutionaryRuleExtractor.class.getResource(
                "/monks/monks-1.test.arff");
        // Instances.main(new String[] {resource.getPath()});
        String[] args = new String[]{
            "-i", // per class statistics
            "-t", resource.getPath(), // train set
            "-T", resource.getPath() // test set (will go with 10-fold CV if not set)
        };
        runClassifier(new CoevolutionaryRuleExtractor(), args);
    }
}
