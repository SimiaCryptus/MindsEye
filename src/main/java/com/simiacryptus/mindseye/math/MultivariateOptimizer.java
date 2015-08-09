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
  public static class Triplet<A,B,C> {
    public final A a;
    public final B b;
    public final C c;
    public Triplet(A a, B b, C c) {
      super();
      this.a = a;
      this.b = b;
      this.c = c;
    }
  }

  static final Logger log = LoggerFactory.getLogger(MultivariateOptimizer.class);

  private final MultivariateFunction f;
  private boolean verbose = false;

  public MultivariateOptimizer(final MultivariateFunction f) {
    this.f = f;
  }
  int maxIterations = 1000;

  public PointValuePair minimize(Pair<double[], Double> last) {
    double dist;
    int iterations = 0;
    do {
      Pair<double[], Double> next = step(last);
      dist = dist(last.getFirst(), next.getFirst());
      last = next;
      if(iterations++>maxIterations){
        throw new RuntimeException("Non convergent");
      }
    } while (dist > 1e-8);
    return eval(last.getFirst());
  }

  public PointValuePair eval(double[] x) {
    return new PointValuePair(x, this.f.value(x));
  }

  public double dist(final double[] last, double[] next) {
    if(isVerbose()) log.debug(String.format("%s -> %s", Arrays.toString(last), Arrays.toString(next)));
    return Math.sqrt(IntStream.range(0, last.length).mapToDouble(i->next[i]-last[i]).map(x->x*x).average().getAsDouble());
  }

  public Pair<double[],Double> step(final Pair<double[], Double> last) {
    //Set<Integer> toMutate = chooseIndexes(1, start.length);
    ArrayList<Integer> l = new ArrayList<>(IntStream.range(0, last.getFirst().length).mapToObj(x->x).collect(Collectors.toList()));
    Collections.shuffle(l);
    Triplet<Integer, double[],Double> opt = IntStream.range(0, last.getFirst().length).parallel().mapToObj(i->{
      MultivariateFunction f2 = Util.kryo().copy(f);
      try {
        PointValuePair minimize = new UnivariateOptimizer(x1 -> {
          return f2.value(copy(last.getFirst(), i, x1));
        }).minimize(last.getFirst()[i]);
        return new Triplet<>(i,copy(last.getFirst(), i, minimize.getFirst()[0]), minimize.getSecond());
      } catch (Exception e) {
        if(verbose) log.debug("Error mutating " + i, e);
        return null;
      }
    })
    .filter(x->null!=x)
    .filter(x->x.c<last.getValue())
    //.min(Comparator.comparing(x->x.c))
    .findFirst()
    .get();
    return new Pair<>(copy(last.getFirst(), opt.a, opt.b[opt.a]), opt.c);
    //return step(start, l.stream().collect(Collectors.toSet()));
  }

  public static double[] copy(final double[] start, int i, double v) {
    double[] next = Arrays.copyOf(start, start.length);
    next[i] = v;
    return next;
  }

  public boolean isVerbose() {
    return verbose;
  }

  public MultivariateOptimizer setVerbose(boolean verbose) {
    this.verbose = verbose;
    return this;
 }

  public PointValuePair minimize(double[] x) {
    return minimize(eval(x));
  }

}