package com.simiacryptus.mindseye.data;

import java.util.Random;
import java.util.function.Function;
import java.util.stream.IntStream;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import com.simiacryptus.mindseye.util.Util;

public final class GaussianDistribution implements Function<Void, double[]> {
  public final MultivariateNormalDistribution distribution;
  private RealMatrix pos;
  private RealMatrix postMult;

  public GaussianDistribution(final int dims) {
    final Random random = Util.R.get();

    final double[] means = new double[dims];
    for (int i = 0; i < means.length; i++)
    {
      means[i] = 1;
    }
    final double[][] diaganals = new double[dims][];
    for (int i = 0; i < diaganals.length; i++)
    {
      diaganals[i] = new double[dims];
      diaganals[i][i] = 1;
    }
    final RandomGenerator rng = new JDKRandomGenerator();
    rng.setSeed(random.nextInt());
    this.distribution = new MultivariateNormalDistribution(rng, means, diaganals);

    this.pos = MatrixUtils.createColumnRealMatrix(IntStream.range(0, dims).mapToDouble(i -> random.nextGaussian() * 0.6).toArray());
    this.postMult = MatrixUtils.createRealMatrix(dims, dims);
    IntStream.range(0, dims).forEach(i -> {
      IntStream.range(0, dims).forEach(j -> {
        this.postMult.setEntry(i, j, random.nextGaussian() * 0.4);
      });
    });
  }

  public GaussianDistribution(final int dims, final double[] pos, final double size) {
    final Random random = Util.R.get();

    final double[] means = new double[dims];
    for (int i = 0; i < means.length; i++)
    {
      means[i] = 1;
    }
    final double[][] diaganals = new double[dims][];
    for (int i = 0; i < diaganals.length; i++)
    {
      diaganals[i] = new double[dims];
      diaganals[i][i] = 1;
    }
    final RandomGenerator rng = new JDKRandomGenerator();
    rng.setSeed(random.nextInt());
    this.distribution = new MultivariateNormalDistribution(rng, means, diaganals);

    this.pos = MatrixUtils.createColumnRealMatrix(pos);
    this.postMult = MatrixUtils.createRealMatrix(dims, dims);
    IntStream.range(0, dims).forEach(i -> {
      IntStream.range(0, dims).forEach(j -> {
        this.postMult.setEntry(i, j, i == j ? size : 0);
      });
    });
  }

  @Override
  public double[] apply(final Void n) {
    final RealMatrix sample = MatrixUtils.createColumnRealMatrix(this.distribution.sample());
    final RealMatrix multiply = this.postMult.multiply(sample);
    final RealMatrix add = multiply.add(this.pos).transpose();
    final double[] ds = add.getData()[0];
    return ds;
  }
}