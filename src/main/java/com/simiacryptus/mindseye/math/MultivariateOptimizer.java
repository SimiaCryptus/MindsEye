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
    return minimize(eval(new double[dims]));
  }
  
  public PointValuePair minimize(PointValuePair initial) {
    int dims = initial.getFirst().length;
    final ThreadLocal<MultivariateFunction> f2 = copyOnFork(f);
    return IntStream.range(0, (int) Math.min(Math.ceil(Math.sqrt(dims)), 2)).limit(1)
        //.parallel()
        .mapToObj(threadNum->{
      final ArrayList<Integer> l = new ArrayList<>(IntStream.range(0, dims).mapToObj(x -> x).collect(Collectors.toList()));
      Collections.shuffle(l);
      PointValuePair accumulator = initial;
      for(int i=0;i<dims;i++){
        Integer d = l.get(i);
        try {
          double[] pos = accumulator.getFirst();
          PointValuePair oneD = new UnivariateOptimizer(x1 -> {
            return f2.get().value(MultivariateOptimizer.copy(pos, d, x1));
          }).minimize();
          accumulator = new PointValuePair(MultivariateOptimizer.copy(pos, d, oneD.getFirst()[0]), oneD.getSecond());
        } catch (final Exception e) {
          if (this.verbose) {
            MultivariateOptimizer.log.debug("Error mutating " + d, e);
          }
        }
      }
      return accumulator;
    }).sorted(Comparator.comparing(p->p.getSecond())).findFirst().orElse(null);
  }
  
  public MultivariateOptimizer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }
  
  public static <T> ThreadLocal<T> copyOnFork(T localValue) {
    final ThreadLocal<T> f2 = new ThreadLocal<T>(){
      @Override
      protected T initialValue() {
        return Util.kryo().copy(localValue);
      }
    };
    f2.set(localValue);
    return f2;
  }
  
}