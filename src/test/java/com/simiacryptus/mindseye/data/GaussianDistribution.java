package com.simiacryptus.mindseye.data;

import java.util.Random;
import java.util.function.Function;
import java.util.stream.IntStream;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import com.simiacryptus.util.Util;

public final class GaussianDistribution implements Function<Void, double[]> {
  private int dims;
  private double[] pos;
  private double size = 1;

  public GaussianDistribution(final int dims) {
    final Random random = Util.R.get();
    this.size = random.nextDouble();
    this.pos = IntStream.range(0, dims).mapToDouble(i -> random.nextGaussian() * 0.6).toArray();
  }

  public GaussianDistribution(final int dims, final double[] pos, final double size) {
    final Random random = Util.R.get();
    this.dims = dims;
    final double[] means = new double[dims];
    for (int i = 0; i < means.length; i++) {
      means[i] = 1;
    }
    final double[][] diaganals = new double[dims][];
    for (int i = 0; i < diaganals.length; i++) {
      diaganals[i] = new double[dims];
      diaganals[i][i] = 1;
    }
    final RandomGenerator rng = new JDKRandomGenerator();
    rng.setSeed(random.nextInt());

    this.pos = pos;
    this.size = size;
  }

  @Override
  public double[] apply(final Void n) {
    final double[] ds = new double[this.dims];
    for (int i = 0; i < this.dims; i++) {
      ds[i] = Util.R.get().nextGaussian() * this.size + this.pos[i];
    }
    return ds;
  }
}
