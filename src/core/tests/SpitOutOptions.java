
package core.tests;

import weka.classifiers.functions.CoevolutionaryRuleExtractor;
import weka.classifiers.functions.EvolutionaryRuleExtractor;

/**
 *
 * @author gmatoga
 */
public class SpitOutOptions {
    public static void main(String[] args) {
        CoevolutionaryRuleExtractor core = new CoevolutionaryRuleExtractor();
        core.setDebug(true);
        core.spitOutOptions();

        EvolutionaryRuleExtractor evo = new EvolutionaryRuleExtractor();
        evo.setDebug(true);
        evo.spitOutOptions();
    }
}
