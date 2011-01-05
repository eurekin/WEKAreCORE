package core.adapters;

import core.DataSetBundle;
import core.ga.RuleChromosomeSignature;
import core.io.dataframe.Row;
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
import java.util.HashSet;
import java.util.List;
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

    public DataAdapter(Instances wekaInsts) {
        // aktualnie intepretowane są String
        // potrzeba tworzyć dataframe opartego o liczby,
        // a do wyświetlania trzymać mapkę z liczb na napisy
        List<DomainMemoizable> l = new ArrayList<DomainMemoizable>();
        List<DomainMemoizable> doms = new ArrayList<DomainMemoizable>();
        List<Column<String>> cols = new ArrayList<Column<String>>();
        for (final Attribute attribute : attrs(wekaInsts)) {
            makeSureIsNominal(attribute);
            doms.add(new AttributeDomain(attribute));
            cols.add(new AttributeColumn(wekaInsts, attribute));
        }
        Attribute classAttribute = wekaInsts.classAttribute();
        makeSureIsNominal(classAttribute);
        DomainMemoizable classDom = new AttributeDomain(classAttribute);
        Column<String> classCol = new AttributeColumn<String>(wekaInsts, classAttribute);
        RuleChromosomeSignature sig = new RuleChromosomeSignature(doms, classDom);
        RuleASCIIPlotter plotter = new RuleASCIIPlotter(sig);

        UniformDataFrame<String, String> udf =
                new UniformDataFrame<String, String>(classCol, cols, wekaInsts.numInstances());
        DataSetBundle bundle = new DataSetBundle(null, plotter, sig, null, null);
    }

    private void makeSureIsNominal(Attribute attribute)
            throws IllegalArgumentException {
        if (!attribute.isNominal()) {
            throw new IllegalArgumentException("Can only handle nominal values");
        }
    }

    private static List<String> values(Attribute i) {
        return list(i.enumerateValues(), String.class);
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

    private static class AttributeColumn<T> extends AbstractColumn<T> {

        public AttributeColumn(Instances inst, Attribute atr) {
            inst.instance(1).stringValue(atr);
            for (Instance i : instances(inst)) {
                list.add((T) i.stringValue(atr));
            }
        }
    }

    private static class AttributeDomain implements DomainMemoizable {

        final HashSet hashSet;

        public AttributeDomain(Attribute attribute) {
            this.hashSet = new HashSet(values(attribute));
        }

        public Set getDomain() {
            return hashSet;
        }

        @Override
        public String toString() {
            return hashSet.toString();
        }
    }
}
