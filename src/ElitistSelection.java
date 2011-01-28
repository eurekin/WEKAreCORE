

import core.ExecutionContextFactory;
import core.adapters.DataAdapter;
import core.adapters.TrainAndTestInstances;
import core.copop.CoPopulations;
import core.copop.PittsIndividual;
import core.ga.Individual;
import core.ga.RulePopulation;
import core.ga.RulePrinter;
import core.ga.ops.ec.ExecutionEnv;
import core.ga.ops.ec.FitnessEval;
import core.ga.ops.ec.FitnessEvaluatorFactory;
import core.utils.ui.ExecutionEnvEditor;
import core.utils.ui.LinearBarGraphPanel;
import core.utils.ui.MMMGraph;
import core.utils.ui.MultiPanelFrame;
import core.vis.RuleASCIIPlotter;
import java.awt.HeadlessException;
import java.util.Random;
import javax.swing.JFrame;
import weka.classifiers.functions.CoevolutionaryRuleExtractor;

/**
 *
 * @author gmatoga
 */
public class ElitistSelection {

    public static void main(String[] args) {
        boolean debug = false;
        if (debug)
            debug();
        else
            compute(System.currentTimeMillis(), false, true);

    }

    public static void debug() {
        boolean lookingForSeed = false;
        if (lookingForSeed)
            for (int i = 1000; i < 10000; i++) {
                System.out.println("Seed == " + i);
                compute(i, true, false);
            }
        else
            compute(1310, true, false);
    }

    public static void compute(long seed, boolean debug, boolean trueRun) {
        FitnessEval fiteval = FitnessEvaluatorFactory.EVAL_FMEASURE;

        TrainAndTestInstances ins = new TrainAndTestInstances("monk-3.train");
        DataAdapter da = new DataAdapter(ins.train());
        ExecutionEnv c = CoevolutionaryRuleExtractor.constructEnvironmentForWEKAInstances(da);
        ExecutionEnv ec = c;


        ec.setRand(new Random(seed));
        ec.setRulePopSize(3);
        int popSize = 3;
        ec.setMaxRuleSetLength(2);
        ec.setRsmp(0.02);
        ec.setMt(0.02);
        ec.setTokenCompetitionEnabled(false);
        ec.setTokenCompetitionWeight(1);
        ec.setEliteSelectionSize(1);
        ec.setRuleSortingEnabled(false);

        if (trueRun) {
            popSize = 200;
            ec.setRulePopSize(200);
            ec.setEliteSelectionSize(1);
            ec.setRsmp(0.02);
            ec.setMt(0.02);
            ec.setRuleSortingEnabled(true);
            ec.setTokenCompetitionEnabled(true);
            ec.setTokenCompetitionWeight(1.0);
            ec.setRand(new Random());
            ec.setMaxRuleSetLength(5);
        }

        if (debug) {
            ec.getDebugOptions().setAllTrue();
            ec.getDebugOptions().setGenerationStatisticsGathered(false);
        }

        CoPopulations co = new CoPopulations(popSize, ec);
        RuleASCIIPlotter plotter = ec.getBundle().getPlotter();
        RulePrinter printer = ec.getBundle().getPrinter();

        MMMGraph graphR = null, graphRS = null;
        if (!debug) {
            graphR = new MMMGraph("Rules");
            graphRS = new MMMGraph("RulesSets");
            constructGUI(graphR, graphRS, ec);
        }



        double maxSoFar = 0, max = 0;
        //      int i = 0;
//        while (maxSoFar <= max || true) {
        for (int i = 0; i < (debug ? 3 : 1000); i++) {
            if (ec.getDebugOptions().isElitistSelectionSpecificOutput()) {
                System.out.println("");
                System.out.println("Generation: " + i);
                System.out.println("Rules:");
                spitOutPops(co, printer);
            }
            co.evolve();
            if (ec.getDebugOptions().isElitistSelectionSpecificOutput()) {
                spitOutPops(co, printer);
                System.out.printf("%d : %s\n", i, co.ruleSetStats());
            } else {
                max = co.ruleSetStats().getMax();
                if (!debug) {
                    graphR.add(co.ruleStats());
                    graphRS.add(co.ruleSetStats());
                }
                if (max < maxSoFar) {
                    System.out.println("****" + i);
                    System.out.println(max);
                    maxSoFar = max;
                }
                if (max > maxSoFar) {
                    maxSoFar = max;
                    System.out.println(max);
                }
            }
            //i++;
        }
    }

    private static void constructGUI(MMMGraph graphR, MMMGraph graphRS, ExecutionEnv ec) throws HeadlessException {
        MultiPanelFrame frame = new MultiPanelFrame();
        frame.getCenterPanel().add(graphR.getChart());
        frame.getCenterPanel().add(graphRS.getChart());
        LinearBarGraphPanel linearBarGraphPanel = new LinearBarGraphPanel();
        frame.setLocationByPlatform(true);
        graphRS.setPanel(linearBarGraphPanel);
        frame.getBottomPanel().add(linearBarGraphPanel);
        frame.setVisible(true);
        JFrame frame2 = new JFrame();
        ExecutionEnvEditor editor = new ExecutionEnvEditor(ec);
        frame2.setLocationByPlatform(true);
        frame2.getContentPane().add(editor);
        frame2.pack();
        frame2.setVisible(true);
    }

    private static void spitOutPops(CoPopulations co, RulePrinter printer) {
        RulePopulation rules = co.rules();
        for (Individual ind : rules) {
            System.out.printf("Fitness=%5.1f%%\n", ind.fitness() * 100);
            System.out.printf("%s  \n", ind.cm());
            String s = printer.prettyPrint(ind.rule());
            System.out.println(s);
            System.out.println("");
        }
        System.out.println("RuleSets");
        for (PittsIndividual pind : co.ruleSets().getIndividuals()) {
            System.out.printf("Fitness=%5.1f%%\n", pind.fitness() * 100);
            System.out.print(pind.getCm() + "   ");
            System.out.println(pind);
            System.out.println(printer.print(pind.getRS()));
        }
    }
}
