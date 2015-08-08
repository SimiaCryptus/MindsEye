package com.simiacryptus.mindseye.test.dev;

import java.util.stream.DoubleStream;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.optim.PointValuePair;
import org.junit.Test;

public class LearningParameterOptimizerDev {
  public static class Opt1 {

    private final UnivariateFunction f;

    public Opt1(UnivariateFunction f) {
      this.f = f;
    }

    public PointValuePair opt() {
      double optimal = 0;
      return new PointValuePair(new double[]{optimal}, f.value(optimal));
    }
    
  }

  public static class OptN {

    private final MultivariateFunction f;

    public OptN(MultivariateFunction f) {
      this.f = f;
    }

    public PointValuePair opt() {
      double[] optimal = new double[]{};
      return new PointValuePair(optimal, f.value(optimal));
    }
    
  }

  @Test
  public void test_univariate(){
    UnivariateFunction f = new UnivariateFunction() {
      @Override
      public double value(double x) {
        return Math.sin(x);
      }
    };
    PointValuePair result = new Opt1(f).opt();
    assert(result.getValue() < 0.99);
  }

  @Test
  public void test_multivariate(){
    MultivariateFunction f = new MultivariateFunction() {
      @Override
      public double value(double[] d) {
        return DoubleStream.of(d).map(x->Math.sin(x)).average().getAsDouble();
      }
    };
    PointValuePair result = new OptN(f).opt();
    assert(result.getValue() < 0.99);
  }
}
