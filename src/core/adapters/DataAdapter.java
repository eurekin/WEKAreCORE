package core.adapters;

import core.DataSetBundle;
import core.ga.RuleChromosomeSignature;
import core.ga.RulePrinter;
import core.io.dataframe.Mapper;
import core.io.dataframe.UniformDataFrame;
import core.io.repr.col.AbstractColumn;
import core.io.repr.col.Column;
import core.io.repr.col.DomainMemoizable;
import core.vis.RuleASCIIPlotter;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

/**
 *
 * @author gmatoga
 */
public class DataAdapter {

    DataSetBundle bundle;

    public DataSetBundle getBundle() {
        return bundle;
    }

    public DataAdapter(Instances wekaInsts) {
        List<DomainMemoizable> doms = new ArrayList<DomainMemoizable>();
        List<Column<Integer>> cols = new ArrayList<Column<Integer>>();
        for (final Attribute attribute : attrs(wekaInsts)) {
            makeSureIsNominal(attribute);
            doms.add(new AttributeDomain(attribute));
            cols.add(new AttributeColumn(wekaInsts, attribute));
        }
        Attribute classAttribute = wekaInsts.classAttribute();
        makeSureIsNominal(classAttribute);
        DomainMemoizable classDom = new AttributeDomain(classAttribute);
        Column<Integer> classCol = new AttributeColumn(wekaInsts, classAttribute);
        System.out.println("Got following cols: " + doms);
        RuleChromosomeSignature sig = new RuleChromosomeSignature(doms, classDom);
        RuleASCIIPlotter plotter = new RuleASCIIPlotter(sig);

        UniformDataFrame<Integer, Integer> udf =
                new UniformDataFrame<Integer, Integer>(
                classCol, cols, wekaInsts.numInstances());

        RulePrinter rp = new RulePrinter(getMapperFor(wekaInsts));
        final String relname = wekaInsts.relationName();
        bundle = new DataSetBundle(udf, plotter, sig, rp, relname);
    }

    private void makeSureIsNominal(Attribute attribute)
            throws IllegalArgumentException {
        if (!attribute.isNominal()) {
            throw new IllegalArgumentException("Can only handle nominal values");
        }
    }

    private static List<Integer> values(Attribute i) {
        return new ArrayList<Integer>(
                toMap(list(i.enumerateValues(), String.class)).keySet());
    }

    private static List<Instance> instances(Instances i) {
        return list(i.enumerateInstances(), Instance.class);
    }

    private static List<Attribute> attrs(Instances i) {
        return list(i.enumerateAttributes(), Attribute.class);
    }

    @SuppressWarnings("unused")
    private static <E> List<E> list(Enumeration en, Class<? extends E> clazz) {
        return Collections.list(en);
    }

    public static void main(String[] args) throws IOException {
        URL resource = DataAdapter.class.getResource("/monks/monks-1.test.arff");

        Instances.main(new String[]{resource.getPath()});

        BufferedReader reader = new BufferedReader(
                new FileReader(resource.getPath()));
        ArffReader arff = new ArffReader(reader);
        Instances data = arff.getData();
        data.setClassIndex(data.numAttributes() - 1);

        DataAdapter dataAdapter = new DataAdapter(data);
    }

    public static Mapper getMapperFor(Instances i) {
        Map<Integer, String> namemap = newmap();
        Map<Integer, Map<Integer, String>> valmap = newmap();
        ArrayList<Attribute> atrs = Collections.list(i.enumerateAttributes());

        int j = 0;
        for (Attribute attribute : atrs) {
            valmap.put(j, toMap(Collections.list(attribute.enumerateValues())));
            namemap.put(j, attribute.name());
            j++;
        }
        String clazzName = i.classAttribute().name();
        Map<Integer, String> classmap =
                toMap(Collections.list(i.classAttribute().enumerateValues()));

        Mapper mapper = new Mapper(valmap, namemap, clazzName, classmap);
        return mapper;
    }

    public static <K, V> HashMap<K, V> newmap() {
        return new HashMap<K, V>();
    }

    public static <V> Map<Integer, V> toMap(List<V> list) {
        Integer i = 0;
        HashMap<Integer, V> map = newmap();
        for (V object : list)
            map.put(i++, object);
        return map;
    }

    private static class AttributeColumn extends AbstractColumn<Integer> {

        public AttributeColumn(Instances inst, Attribute atr) {
            for (Instance i : instances(inst)) {
                list.add((int) (i.value(atr)));
            }
        }
    }

    private static class AttributeDomain implements DomainMemoizable<Integer> {

        final HashSet<Integer> hashSet;

        public AttributeDomain(Attribute attribute) {
            this.hashSet = new HashSet<Integer>(values(attribute));
        }

        public Set<Integer> getDomain() {
            return hashSet;
        }

        @Override
        public String toString() {
            return hashSet.toString();
        }
    }
}
