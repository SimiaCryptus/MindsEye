package com.simiacryptus.mindseye.math;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.util.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;

public class MultivariateOptimizer {
  public static class Triplet<A, B, C> {
    public final A a;
    public final B b;
    public final C c;
    
    public Triplet(final A a, final B b, final C c) {
      super();
      this.a = a;
      this.b = b;
      this.c = c;
    }
  }
  
  static final Logger log = LoggerFactory.getLogger(MultivariateOptimizer.class);
  
  public static double[] copy(final double[] start, final int i, final double v) {
    final double[] next = Arrays.copyOf(start, start.length);
    next[i] = v;
    return next;
  }
  
  private final MultivariateFunction f;
  
  int maxIterations = 1000;
  private boolean verbose = false;
  
  public MultivariateOptimizer(final MultivariateFunction f) {
    this.f = f;
  }
  
  public double dist(final double[] last, final double[] next) {
    if (isVerbose()) {
      MultivariateOptimizer.log.debug(String.format("%s -> %s", Arrays.toString(last), Arrays.toString(next)));
    }
    return Math.sqrt(IntStream.range(0, last.length).mapToDouble(i -> next[i] - last[i]).map(x -> x * x).average().getAsDouble());
  }
  
  public PointValuePair eval(final double[] x) {
    return new PointValuePair(x, this.f.value(x));
  }
  
  public boolean isVerbose() {
    return this.verbose;
  }
  
  public PointValuePair minimize(int dims) {
    double[] start = new double[dims];
    return minimize(eval(start));
  }
  
  public PointValuePair minimize(Pair<double[], Double> last) {
    double dist;
    int iterations = 0;
    do {
      final Pair<double[], Double> next = step(last);
      dist = dist(last.getFirst(), next.getFirst());
      last = next;
      if (iterations++ > this.maxIterations) throw new RuntimeException("Non convergent");
    } while (dist > 1e-8);
    if (isVerbose()) log.debug(String.format("Result: %s with prev dist %s", Arrays.toString(last.getFirst()), dist));
    return eval(last.getFirst());
  }
  
  public MultivariateOptimizer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }
  
  public Pair<double[], Double> step(final Pair<double[], Double> initial) {
    // Set<Integer> toMutate = chooseIndexes(1, start.length);
    final ArrayList<Integer> l = new ArrayList<>(IntStream.range(0, initial.getFirst().length).mapToObj(x -> x).collect(Collectors.toList()));
    Collections.shuffle(l);
    return IntStream.range(0, initial.getFirst().length).parallel().mapToObj(i -> {
      final MultivariateFunction f2 = Util.kryo().copy(this.f);
      try {
        double start = initial.getFirst()[i];
        final PointValuePair minimize = new UnivariateOptimizer(x1 -> {
          return f2.value(MultivariateOptimizer.copy(initial.getFirst(), i, x1));
        }).minimize(0.==start?1.:start);
        return new Triplet<>(i, MultivariateOptimizer.copy(initial.getFirst(), i, minimize.getFirst()[0]), minimize.getSecond());
      } catch (final Exception e) {
        if (this.verbose) {
          MultivariateOptimizer.log.debug("Error mutating " + i, e);
        }
        return null;
      }
    })
      .filter(x -> null != x)
      .filter(x -> x.c < initial.getValue())
      .limit(2)
      .min(Comparator.comparing(x -> x.c))
      .map(opt->new Pair<>(MultivariateOptimizer.copy(initial.getFirst(), opt.a, opt.b[opt.a]), opt.c))
      .orElse(initial);
  }
  
}