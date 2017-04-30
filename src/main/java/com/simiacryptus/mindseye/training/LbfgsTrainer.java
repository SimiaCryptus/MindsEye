package com.simiacryptus.mindseye.training;

import com.simiacryptus.mindseye.net.DeltaBuffer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Collectors;

public class LbfgsTrainer extends LineSearchTrainer {
  private int minHistory = 2;

  protected List<DeltaBuffer> getDirection(List<DeltaBuffer> startGradient) {
    // See also https://papers.nips.cc/paper/5333-large-scale-l-bfgs-using-mapreduce
    List<DeltaBuffer> descent = startGradient.stream().map(x -> x.scale(-1)).collect(Collectors.toList());
    if(history.size() > minHistory) {
      List<double[]> p = descent.stream().map(x -> x.copyDelta()).collect(Collectors.toList());
      double[] alphas = new double[history.size()];
      for (int i = history.size() - 2; i >= 0; i--) {
        List<double[]> si = minus(history.get(i + 1).targetSet, history.get(i).targetSet);
        List<double[]> yi = minus(history.get(i + 1).deltaSet, history.get(i).deltaSet);
        alphas[i] = dot(si, p) / dot(si, yi);
        p = minus(p, multiply(yi, alphas[i]));
      }
      List<double[]> sk1 = minus(history.get(history.size() - 1).targetSet, history.get(history.size() - 2).targetSet);
      List<double[]> yk1 = minus(history.get(history.size() - 1).deltaSet, history.get(history.size() - 2).deltaSet);
      p = multiply(p, dot(sk1, yk1) / dot(yk1, yk1));
      for (int i = 0; i < history.size() - 1; i++) {
        List<double[]> si = minus(history.get(i + 1).targetSet, history.get(i).targetSet);
        List<double[]> yi = minus(history.get(i + 1).deltaSet, history.get(i).deltaSet);
        double beta = dot(yi, p) / dot(si, yi);
        p = add(p, multiply(si, alphas[i] - beta));
      }
      List<double[]> _p = p;
      for (int i = 0; i < descent.size(); i++) {
        int _i = i;
        Arrays.setAll(descent.get(i).delta, j -> _p.get(_i)[j]);
      }
    }
    return descent;
  }

  private List<double[]> minus(List<double[]> a, List<double[]> b) {
    return op(a, b, (x, y) -> x - y);
  }

  private List<double[]> add(List<double[]> a, List<double[]> b) {
    return op(a, b, (x, y) -> x + y);
  }

  private double dot(List<double[]> a, List<double[]> b) {
    return sum(multiply(a, b));
  }

  private List<double[]> multiply(List<double[]> a, List<double[]> b) {
    return op(a, b, (x, y) -> x * y);
  }

  private List<double[]> multiply(List<double[]> a, double b) {
    return op(a, x -> x * b);
  }

  private double sum(List<double[]> a) {
    return a.stream().mapToDouble(x -> Arrays.stream(x).sum()).sum();
  }

  private List<double[]> op(List<double[]> a, List<double[]> b, DoubleBinaryOperator fn) {
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

  private List<double[]> op(List<double[]> a, DoubleUnaryOperator fn) {
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

  public int getMinHistory() {
    return minHistory;
  }

  public LbfgsTrainer setMinHistory(int minHistory) {
    this.minHistory = minHistory;
    return this;
  }
}
