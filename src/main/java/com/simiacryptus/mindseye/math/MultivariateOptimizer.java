package com.simiacryptus.mindseye.math;

import java.util.Arrays;
import java.util.stream.IntStream;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.optim.PointValuePair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MultivariateOptimizer {
  static final Logger log = LoggerFactory.getLogger(MultivariateOptimizer.class);

  private final MultivariateFunction f;
  private boolean verbose = true;

  public MultivariateOptimizer(final MultivariateFunction f) {
    this.f = f;
  }

  public PointValuePair minimize(double[] last) {
    double dist;
    do {
      double[] next = step(last);
      dist = dist(last, next);
      last = next;
    } while (dist > 1e-8);
    return new PointValuePair(last, this.f.value(last));
  }

  public double dist(final double[] last, double[] next) {
    if(isVerbose()) log.debug(String.format("%s -> %s", Arrays.toString(last), Arrays.toString(next)));
    return Math.sqrt(IntStream.range(0, last.length).mapToDouble(i->next[i]-last[i]).map(x->x*x).average().getAsDouble());
  }

  public double[] step(final double[] start) {
    return IntStream.range(0, start.length).parallel().mapToDouble(i->{
      return new UnivariateOptimizer(x->{
        double[] pt = Arrays.copyOf(start, start.length);
        pt[i] = x;
        return f.value(pt);
      }).minimize(start[i]).getFirst()[0];
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