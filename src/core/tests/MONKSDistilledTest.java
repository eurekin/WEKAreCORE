package core.tests;

import core.adapters.DataAdapter;
import core.adapters.TrainAndTestInstances;
import core.copop.RuleSet;
import core.ga.Evaluator;
import core.ga.ops.ec.ExecutionEnv;
import core.ga.ops.ec.FitnessEval;
import core.io.dataframe.Row;
import core.stat.BinaryConfMtx;
import core.stat.ConfMtx;
import core.vis.RuleASCIIPlotter;
import weka.classifiers.functions.CoevolutionaryRuleExtractor;
import static core.mock.FluentBulders.*;

/**
 *
 * @author gmatoga
 */
public class MONKSDistilledTest {

    TrainAndTestInstances monk = new TrainAndTestInstances("monks-3.test");
    DataAdapter adapt = new DataAdapter(monk.test());

    public void test() {
        ExecutionEnv ec = CoevolutionaryRuleExtractor.constructEnvironmentForWEKAInstances(adapt);
        Evaluator evl = ec.evaluator();
        ConfMtx cm = new ConfMtx(2);
        // 0 - tło
        // 1 - koncept
        RuleSet rs = makeRuleSet(0,
                rule().X().X().X().in(0).in(2).X().clazz(1),
                rule().X().out(2).X().X().out(3).X().clazz(1));

        System.out.println("Evaluator class = "  + evl.getClass());
        for (Row row : adapt.getBundle().getData()) {
            evl.evaluate(rs, row, cm);
        }

        RuleASCIIPlotter plotter = ec.getBundle().getPlotter();
        CoevolutionaryRuleExtractor.visualizeData(monk.test(), ec, plotter);
        plotter.detailedPlots(rs);
        System.out.println(cm);

        System.out.println("Per class fitness");
        FitnessEval FitnessEval = ec.fitnessEvaluator();
        for (BinaryConfMtx bcm : cm.getCMes()) {
            System.out.println(FitnessEval.eval(bcm));
        }


    }

    public static void main(String[] args) {

        new MONKSDistilledTest().test();
    }
}
