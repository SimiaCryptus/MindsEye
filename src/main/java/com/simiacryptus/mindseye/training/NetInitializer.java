package com.simiacryptus.mindseye.training;

import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Coordinate;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;

public class NetInitializer {
  private static final Logger log = LoggerFactory.getLogger(NetInitializer.class);

  private double amplitude = 0.;
  private boolean verbose;

  private int initialize(final BiasLayer l) {
    final double[] a = l.bias;
    final Random random = Util.R.get();
    int sum = 0;
    for (int i = 0; i < a.length; i++) {
      final double prev = a[i];
      final double prevEntropy = entropy(l);
      a[i] = randomWeight(l, random);
      final double nextEntropy = entropy(l);
      if (nextEntropy < prevEntropy) {
        a[i] = prev;
      } else {
        sum += 1;
      }
    }
    return sum;
  }

  private int initialize(final DenseSynapseLayer l) {
    final double[] a = l.weights.getData();
    final Random random = Util.R.get();
    return l.weights.coordStream().mapToInt(idx -> {
      final int i = idx.index;
      final double prev = a[i];
      final double prevEntropy = entropy(l, idx);
      a[i] = randomWeight(l, random);
      final double nextEntropy = entropy(l, idx);
      if (nextEntropy < prevEntropy) {
        a[i] = prev;
        return 0;
      } else
        return 1;
    }).sum();
  }

  public void initialize(NNLayer<DAGNetwork> net) {
    if (this.verbose) {
      log.debug(String.format("Initialize %s", net));
    }
    if(this.getAmplitude()<=0.) return;
    final List<NNLayer<?>> layers = net.getChildren();
    for (int i = 0; i < 5; i++) {
      layers.stream().filter(l -> (l instanceof DenseSynapseLayer)).map(l -> (DenseSynapseLayer) l).filter(l -> !l.isFrozen()).forEach(this::initialize);
      layers.stream().filter(l -> (l instanceof BiasLayer)).map(l -> (BiasLayer) l).filter(l -> !l.isFrozen()).forEach(this::initialize);
    }
  }

  protected double randomWeight(final BiasLayer l, final Random random) {
    return this.getAmplitude() * random.nextGaussian() * 0.2;
  }

  protected double randomWeight(final DenseSynapseLayer l, final Random random) {
    return this.getAmplitude() * random.nextGaussian() / Math.sqrt(l.weights.getDims()[0]);
  }

  private double entropy(final BiasLayer l) {
    return 0;
  }

  private double entropy(final DenseSynapseLayer l, final Coordinate idx) {
    final NDArray weights = l.weights;
    final int[] dims = weights.getDims();
    final int columns = dims[0];
    final int rows = dims[1];
    final DoubleMatrix matrix = new DoubleMatrix(columns, rows, weights.getData()).transpose();
    // DoubleMatrix matrix = new DoubleMatrix(rows, columns,
    // l.weights.getData());
    return IntStream.range(0, rows).filter(i -> i == idx.coords[1]).mapToDouble(i -> i).flatMap(i -> {
      return IntStream.range(0, rows).mapToDouble(j -> {
        final ArrayRealVector vi = new ArrayRealVector(matrix.getRow((int) i).toArray());
        if (vi.getNorm() <= 0.)
          return 0.;
        vi.unitize();
        final ArrayRealVector vj = new ArrayRealVector(matrix.getRow(j).toArray());
        if (vj.getNorm() <= 0.)
          return 0.;
        vj.unitize();
        return Math.acos(vi.cosine(vj));
      });
    }).average().getAsDouble();
  }

  public double getAmplitude() {
    return amplitude;
  }

  public void setAmplitude(double initAmplitude) {
    this.amplitude = initAmplitude;
  }

}
