package com.simiacryptus.mindseye.math;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.optim.PointValuePair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.util.Util;

/**
 * Specialized multivariate optimizer for discovering a learning rate
 * metaparameter vector. Assumes all rates should be in [0,max). Assumes a very
 * high rate of exclusionary interdependence. Assumes null vector and solutions
 * beyond threshold magnitude are invalid Assumes approximately parabolic
 * optimum basin Specializes in discovering optimal metaparameters over wide
 * range of scales eg [1e-5,1e5].
 *
 * @author Andrew Charneski
 */
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

  private static double[] copy(final double[] start, final int i, final double v) {
    final double[] next = Arrays.copyOf(start, start.length);
    next[i] = v;
    return next;
  }

  private final MultivariateFunction f;
  int maxIterations = 1000;
  private double maxRate = 1e5;
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

  public double getMaxRate() {
    return this.maxRate;
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  public PointValuePair minimize(final int dims) {
    return minimize(eval(new double[dims]));
  }

  public PointValuePair minimize(final PointValuePair initial) {
    final int dims = initial.getFirst().length;
    final ThreadLocal<MultivariateFunction> f2 = Util.copyOnFork(this.f);
    IntStream dimensionStream = IntStream.range(0, (int) Math.min(Math.ceil(Math.sqrt(dims)), 1)).parallel();
    Stream<ArrayList<Integer>> dimensionalPermuation = dimensionStream.mapToObj(threadNum -> {
      final ArrayList<Integer> l = new ArrayList<>(IntStream.range(0, dims).mapToObj(x -> x).collect(Collectors.toList()));
      Collections.shuffle(l);
      return l;
    }).distinct();
    return dimensionalPermuation.map(l -> {
      PointValuePair accumulator = initial;
      for (int i = 0; i < dims; i++) {
        final Integer d = l.get(i);
        accumulator = optimizeVariable(f2, accumulator, d);
      }
      return accumulator;
    }).sorted(Comparator.comparing(p -> p.getSecond())).findFirst().orElse(null);
  }

  private PointValuePair optimizeVariable(final ThreadLocal<MultivariateFunction> f, final PointValuePair accumulator, final int dimension) {
    final double[] prevVars = accumulator.getFirst();
    try {
      final PointValuePair oneD = new UnivariateOptimizer(x1 -> {
        return f.get().value(MultivariateOptimizer.copy(prevVars, dimension, x1));
      }).setMaxRate(getMaxRate()).minimize();
      double[] nextVars = MultivariateOptimizer.copy(prevVars, dimension, oneD.getFirst()[0]);
      return new PointValuePair(nextVars, oneD.getSecond());
    } catch (final Throwable e) {
      if (isVerbose()) {
        MultivariateOptimizer.log.debug("Error optimizing dimension " + dimension, e);
      }
      return accumulator;
    }
  }

  public MultivariateOptimizer setMaxRate(final double maxRate) {
    this.maxRate = maxRate;
    return this;
  }

  public MultivariateOptimizer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

}
