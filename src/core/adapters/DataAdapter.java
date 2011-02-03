package core.adapters;

import core.io.repr.col.IntegerDomain;
import core.DataSetBundle;
import core.ga.RuleChromosomeSignature;
import core.ga.RulePrinter;
import core.io.dataframe.Mapper;
import core.io.dataframe.DataFrame;
import core.io.repr.col.AbstractColumn;
import core.io.repr.col.Column;
import core.io.repr.col.Domain;
import core.io.repr.col.DomainMemoizable;
import core.io.repr.col.FloatDomain;
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
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import core.vis.RuleASCIIPlotter;

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
        List<Column> cols = new ArrayList<Column>();
        for (final Attribute attribute : attrs(wekaInsts)) {
            makeSureIsNumericalOrNominal(attribute);
            addThisToThat(doms, attribute, wekaInsts);
            cols.add(new AttributeColumn(wekaInsts, attribute));
        }
        Attribute classAttribute = wekaInsts.classAttribute();
        makeSureIsNominal(classAttribute);
        DomainMemoizable classDom = new AttributeDomain(classAttribute);
        Column classCol = new AttributeColumn(wekaInsts, classAttribute);
        RuleChromosomeSignature sig = new RuleChromosomeSignature(doms, classDom);

        // TODO XXX temporarily disabled
        RuleASCIIPlotter plotter = null;
        try {
            plotter = new RuleASCIIPlotter(sig);
        } catch (Exception e) {
            e.printStackTrace();
        }
        DataFrame udf = new DataFrame(
                classCol, cols, wekaInsts.numInstances());

        // TODO XXX temporarily disabled
        RulePrinter rp = null;
        try {
            rp = new RulePrinter(getMapperFor(wekaInsts));
        } catch (Exception e) {
            e.printStackTrace();
        }
        final String relname = wekaInsts.relationName();
        bundle = new DataSetBundle(udf, plotter, sig, rp, relname);
    }

    private void addThisToThat(List<DomainMemoizable> doms,
            final Attribute attribute, Instances inst) {
        if (attribute.isNominal()) {
            doms.add(new AttributeDomain(attribute));
        } else if (attribute.isNumeric()) {
            doms.add(new RealDomain(attribute, inst));
        } else
            throw new IllegalStateException("Unrecognized attribute type");
    }

    private void makeSureIsNumericalOrNominal(Attribute attribute) {
        if (!(attribute.isNominal() || attribute.isNumeric())) {
            throw new IllegalArgumentException(
                    "Can only handle nominal or numeric values");
        }
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
            if (attribute.isNominal()) {
                valmap.put(j, toMap(Collections.list(attribute.enumerateValues())));
            } else {
                valmap.put(j, null);
            }
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

    private static class AttributeColumn extends AbstractColumn {

        public AttributeColumn(Instances inst, Attribute atr) {
            for (Instance i : instances(inst)) {
                list.add((float) (i.value(atr)));
            }
        }
    }

    private static class RealDomain implements DomainMemoizable {

        private Domain domain;
        float max = -Float.MAX_VALUE;
        float min = Float.MAX_VALUE;

        public RealDomain(Attribute attribute, Instances inst) {
            findMinMax(attribute, inst);
            //System.out.printf("Adding numeric domain in bounds <%f, %f>\n", min, max);
            domain = new FloatDomain(min, max);
        }

        private void findMinMax(Attribute atr, Instances inst) {
            for (int i = 0; i < inst.numInstances(); i++) {
                Instance in = inst.instance(i);
                float v = (float) in.value(atr);
                if (v < min)
                    min = v;
                if (v > max)
                    max = v;
            }
        }

        public Domain getDomain() {
            return domain;
        }
    }

    private static class AttributeDomain implements DomainMemoizable {

        final IntegerDomain domain;

        public AttributeDomain(Attribute attribute) {
            HashSet<Integer> hashSet = new HashSet<Integer>(values(attribute));
            this.domain = new IntegerDomain(hashSet);
        }

        public Domain getDomain() {
            return domain;
        }

        @Override
        public String toString() {
            return domain.toString();
        }
    }
}
