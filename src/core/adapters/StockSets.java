package core.adapters;

/**
 *
 * @author gmatoga
 */
public class StockSets {

    public static final FilterableDataSet SETS = new FilterableDataSet() {

        {
            add(new TrainAndTestInstances("iris"));
            add(new TrainAndTestInstances("glass"));
            add(new TrainAndTestInstances("diabetes"));
            //add(new TrainAndTestInstances("soybean"));
            add(new TrainAndTestInstances("monks-1.train", "monks-1.test"));
            add(new TrainAndTestInstances("monks-2.train", "monks-2.test"));
            add(new TrainAndTestInstances("monks-3.train", "monks-3.test"));
        }
    };

    public static void main(String[] args) {
        for (TrainAndTestInstances object : SETS.noMissing()) {
            System.out.println(object.test().relationName());
        }
    }

    public static TrainAndTestInstances monks1() {
        return new TrainAndTestInstances("monks-1.train", "monks-1.test");
    }

    public static TrainAndTestInstances monks2() {
        return new TrainAndTestInstances("monks-2.train", "monks-2.test");
    }

    public static TrainAndTestInstances monks3() {
        return new TrainAndTestInstances("monks-3.train", "monks-3.test");
    }
}
