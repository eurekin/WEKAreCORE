package weka.classifiers.functions;


import core.evo.EvolutionPopulation;


/**
 *
 * @author gmatoga
 */
public interface EvolutionCallback {

    public void coevolutionCallback(EvolutionPopulation pop);
}
