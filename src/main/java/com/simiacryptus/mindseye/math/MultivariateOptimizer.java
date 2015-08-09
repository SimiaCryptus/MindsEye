package com.simiacryptus.mindseye.math;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.util.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;

public class MultivariateOptimizer {
  static final Logger log = LoggerFactory.getLogger(MultivariateOptimizer.class);

  private final MultivariateFunction f;
  private boolean verbose = false;

  public MultivariateOptimizer(final MultivariateFunction f) {
    this.f = f;
  }
  int maxIterations = 1000;

  public PointValuePair minimize(double[] last) {
    double dist;
    int iterations = 0;
    do {
      double[] next = step(last);
      dist = dist(last, next);
      last = next;
      if(iterations++>maxIterations){
        throw new RuntimeException("Non convergent");
      }
    } while (dist > 1e-8);
    return new PointValuePair(last, this.f.value(last));
  }

  public double dist(final double[] last, double[] next) {
    if(isVerbose()) log.debug(String.format("%s -> %s", Arrays.toString(last), Arrays.toString(next)));
    return Math.sqrt(IntStream.range(0, last.length).mapToDouble(i->next[i]-last[i]).map(x->x*x).average().getAsDouble());
  }

  public double[] step(final double[] start) {
    //Set<Integer> toMutate = chooseIndexes(1, start.length);
    ArrayList<Integer> l = new ArrayList<>(IntStream.range(0, start.length).mapToObj(x->x).collect(Collectors.toList()));
    Collections.shuffle(l);
    Pair<Integer, double[]> opt = l.stream().map(i->{
      try {
        return new Pair<>(i,step(start, new HashSet<Integer>(Arrays.asList(i))));
      } catch (Exception e) {
        if(verbose) log.debug("Error mutating " + i, e);
        return null;
      }
    }).filter(x->null!=x).findFirst().get();
    double[] next = Arrays.copyOf(start, start.length);
    next[opt.getFirst()] = opt.getSecond()[opt.getFirst()];
    return next;
    //return step(start, l.stream().collect(Collectors.toSet()));
  }

  public Set<Integer> chooseIndexes(int numberOfSelections, int maxValue) {
    Set<Integer> toMutate = IntStream.generate(()->(int)(maxValue*Math.random())).distinct().limit(numberOfSelections).mapToObj(x->x).collect(Collectors.toSet());
    return toMutate;
  }

  public double[] step(final double[] start, Set<Integer> toMutate) {
    return IntStream.range(0, start.length).parallel().mapToDouble(i->{
      return toMutate.contains(i)?new UnivariateOptimizer(x->{
        double[] pt = Arrays.copyOf(start, start.length);
        pt[i] = x;
        return f.value(pt);
      }).minimize(start[i]).getFirst()[0]:start[i];
    }).toArray();
  }

  public boolean isVerbose() {
    return verbose;
  }

  public MultivariateOptimizer setVerbose(boolean verbose) {
    this.verbose = verbose;
    return this;
 }

}