package com.simiacryptus.mindseye.data;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

import org.apache.commons.math3.analysis.interpolation.LoessInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;

import com.simiacryptus.mindseye.util.Util;

public final class SnakeDistribution implements Function<Void, double[]> {
  public final int dims;
  public final int nodes;
  public final List<PolynomialSplineFunction> parametricFuctions;
  public final double radius;

  public SnakeDistribution(final int dims, final Random random) {
    this(dims, random, 10, 0.01);
  }

  public SnakeDistribution(final int dims, final Random random, final int nodes) {
    this(dims, random, nodes, 0.01);
  }

  public SnakeDistribution(final int dims, final Random random, final int nodes, final double radius) {
    this.dims = dims;
    this.radius = radius;
    this.nodes = nodes;
    final List<PolynomialSplineFunction> parametricFuctions = new ArrayList<PolynomialSplineFunction>();
    for (int j = 0; j < dims; j++) {
      final double[] xval = new double[nodes];
      final double[] yval = new double[nodes];
      for (int i = 0; i < xval.length; i++) {
        xval[i] = random.nextDouble() * 4 - 2;
        yval[i] = i * 1. / (xval.length - 1);
      }
      parametricFuctions.add(new LoessInterpolator().interpolate(yval, xval));
    }

    this.parametricFuctions = parametricFuctions;
  }

  @Override
  public double[] apply(final Void n) {
    final Random random = Util.R.get();
    final double[] pt = new double[this.dims];
    final double t = random.nextDouble();
    for (int i = 0; i < pt.length; i++) {
      pt[i] = this.parametricFuctions.get(i).value(t);
      pt[i] += random.nextGaussian() * this.radius;
    }
    return pt;
  }

  public int getDimension() {
    return this.dims;
  }

}
