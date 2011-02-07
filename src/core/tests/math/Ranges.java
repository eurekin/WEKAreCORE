package core.tests.math;

import java.util.ArrayList;
import java.util.Arrays;

/**
 *
 * @author gmatoga
 */
public class Ranges {

    public static ArrayList<Double> logrange(double start, double end, int steps) {
        return new ArrayList<Double>(Arrays.asList(box( // boilerplate
                transform(
                range(start, end, steps),
                new Pow(10d)))));
    }

    public static Double[] box(double[] what) {
        Double[] boxed = new Double[what.length];
        for (int i = 0; i < boxed.length; i++)
            boxed[i] = what[i];
        return boxed;
    }

    public static double interpolate(double start, double val, double end) {
        final double range = end - start;
        return start + range * val;
    }

    public static double[] range(double start, double end, int steps) {
        double[] range = new double[steps];
        for (int i = 0; i < steps; i++) {
            range[i] = interpolate(start, ((double) i / (steps - 1)), end);
        }
        return range;
    }

    public static double[] transform(double[] what, Transformer how) {
        double[] range = new double[what.length];
        for (int i = 0; i < range.length; i++)
            range[i] = how.transform(what[i]);
        return range;
    }

    public static interface Transformer {

        public double transform(double d);
    }

    public static class Pow implements Transformer {

        private final double a;

        public Pow(double a) {
            this.a = a;
        }

        public double transform(double d) {
            return Math.pow(a, d);
        }
    }
}
