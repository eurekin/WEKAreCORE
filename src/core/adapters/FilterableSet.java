package core.adapters;

import java.util.HashSet;

/**
 *
 * @author gmatoga
 */
public class FilterableSet<T> extends HashSet<T> {

    public FilterableSet<T> filter(SetFilter filter) {
        FilterableSet<T> filtered = new FilterableSet<T>();
        for (T set : this) {
            if (filter.include(set))
                filtered.add(set);
        }
        return filtered;
    }
}
