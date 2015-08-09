package com.simiacryptus.mindseye.test.dev;

import java.util.Arrays;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.optim.PointValuePair;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.math.MultivariateOptimizer;
import com.simiacryptus.mindseye.math.UnivariateOptimizer;

public class LearningParameterOptimizerDev {
  static final Logger log = LoggerFactory.getLogger(LearningParameterOptimizerDev.class);
  
  public static double[] ones(final int dims) {
    final double[] last = DoubleStream.generate(() -> 1.).limit(dims).toArray();
    return last;
  }
  
  @Test
  public void test_multivariate() {
    final double offset = 0.4;
    final MultivariateFunction f = new MultivariateFunction() {
      @Override
      public double value(final double[] a) {
        final double t = IntStream.range(0, a.length).mapToDouble(i -> {
          final double x = a[i];
          final double r = -x * Math.sin(x + offset) + 0.4 * Math.sin((1 + i) * 100 * x);
          return r;
        }).average().getAsDouble();
        LearningParameterOptimizerDev.log.debug(String.format("%s -> %s", Arrays.toString(a), t));
        return t;
      }
    };
    final PointValuePair result = new MultivariateOptimizer(f).minimize(LearningParameterOptimizerDev.ones(3));
    assert result.getValue() < -0.99;
    LearningParameterOptimizerDev.log.debug(String.format("%s -> %s", Arrays.toString(result.getFirst()), result.getSecond()));
  }
  
  @Test
  public void test_univariate() {
    final double offset = .5;
    final UnivariateFunction f = new UnivariateFunction() {
      @Override
      public double value(final double x) {
        final double r = -Math.sin(x + offset) + 0.4 * Math.sin(100 * x);
        LearningParameterOptimizerDev.log.debug(String.format("%s -> %s", x, r));
        return r;
      }
    };
    final PointValuePair result = new UnivariateOptimizer(f).minimize();
    assert result.getValue() < -0.99;
  }
}
