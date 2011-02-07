package core.adapters;

/**
 *
 * @author gmatoga
 */
public class FilterableDataSet extends FilterableSet<TrainAndTestInstances> {

    @Override
    public FilterableDataSet filter(SetFilter filter) {
        FilterableDataSet filtered = new FilterableDataSet();
        for (TrainAndTestInstances set : this) {
            if (filter.include(set))
                filtered.add(set);
        }
        return filtered;
    }

    public FilterableDataSet noMissing() {
        return filter(NO_MISSING_VALUES);
    }

    public FilterableDataSet onlyNominal() {
        return filter(ONLY_NOMINAL);
    }

    public FilterableDataSet onlyNumeric() {
        return filter(ONLY_NUMERIC);
    }

    public FilterableDataSet noCrossvalidation() {
        return filter(ONLY_TESTTRAINDIFF);
    }

    public FilterableDataSet onlyCrossvalidation() {
        return filter(negateFilter);
    }
    public static final SetFilter<TrainAndTestInstances> ONLY_NOMINAL =
            new SetFilter<TrainAndTestInstances>() {

                public boolean include(TrainAndTestInstances set) {
                    return set.onlyNominal();
                }
            };
    public static final SetFilter<TrainAndTestInstances> ONLY_NUMERIC =
            new SetFilter<TrainAndTestInstances>() {

                public boolean include(TrainAndTestInstances set) {
                    return set.onlyNumeric();
                }
            };
    public static final SetFilter<TrainAndTestInstances> NO_MISSING_VALUES =
            new SetFilter<TrainAndTestInstances>() {

                public boolean include(TrainAndTestInstances set) {
                    return !set.hasMissing();
                }
            };
    public static final SetFilter<TrainAndTestInstances> ONLY_TESTTRAINDIFF =
            new SetFilter<TrainAndTestInstances>() {

                public boolean include(TrainAndTestInstances set) {
                    return set.testDifferentFromTrain();
                }
            };

    public static final class NegateFilter<T> implements SetFilter<T> {

        private final SetFilter<T> toNegate;

        public NegateFilter(SetFilter<T> toNegate) {
            this.toNegate = toNegate;
        }

        public boolean include(T set) {
            return !toNegate.include(set);
        }
    }
    private static final NegateFilter negateFilter =
            new NegateFilter(ONLY_TESTTRAINDIFF);
}
