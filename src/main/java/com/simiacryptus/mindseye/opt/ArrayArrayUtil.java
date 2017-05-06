package com.simiacryptus.mindseye.opt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * Created by Andrew Charneski on 5/6/2017.
 */
public class ArrayArrayUtil {

    public static List<double[]> minus(List<double[]> a, List<double[]> b) {
        return op(a, b, (x, y) -> x - y);
    }

    public static List<double[]> add(List<double[]> a, List<double[]> b) {
        return op(a, b, (x, y) -> x + y);
    }

    public static double dot(List<double[]> a, List<double[]> b) {
        return sum(multiply(a, b));
    }

    public static List<double[]> multiply(List<double[]> a, List<double[]> b) {
        return op(a, b, (x, y) -> x * y);
    }

    public static List<double[]> multiply(List<double[]> a, double b) {
        return op(a, x -> x * b);
    }

    public static double sum(List<double[]> a) {
        return a.stream().mapToDouble(x -> Arrays.stream(x).sum()).sum();
    }

    public static List<double[]> op(List<double[]> a, List<double[]> b, DoubleBinaryOperator fn) {
        assert (a.size() == b.size());
        ArrayList<double[]> list = new ArrayList<>();
        for (int i = 0; i < a.size(); i++) {
            assert (a.get(i).length == b.get(i).length);
            double[] c = new double[a.get(i).length];
            for (int j = 0; j < a.get(i).length; j++) {
                c[j] = fn.applyAsDouble(a.get(i)[j], b.get(i)[j]);
            }
            list.add(c);
        }
        return list;

    }

    public static List<double[]> op(List<double[]> a, DoubleUnaryOperator fn) {
        ArrayList<double[]> list = new ArrayList<>();
        for (int i = 0; i < a.size(); i++) {
            double[] c = new double[a.get(i).length];
            for (int j = 0; j < a.get(i).length; j++) {
                c[j] = fn.applyAsDouble(a.get(i)[j]);
            }
            list.add(c);
        }
        return list;
    }

}
