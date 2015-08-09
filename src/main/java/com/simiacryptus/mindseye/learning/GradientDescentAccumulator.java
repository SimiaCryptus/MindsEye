package com.simiacryptus.mindseye.learning;

import java.util.Arrays;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;

public class GradientDescentAccumulator {
  public static boolean DEBUG = false;
  private static final Logger log = LoggerFactory.getLogger(GradientDescentAccumulator.class);

  private final DeltaSink sink;

  protected GradientDescentAccumulator() {
    super();
    this.sink = null;
  }

  public GradientDescentAccumulator(final DeltaSink sink) {
    this.sink = sink;
  }

  public void feed(final NDArray weightGradient, final double[] data) {
    if (null == weightGradient) throw new IllegalArgumentException();
    if (null == data) throw new IllegalArgumentException();
    if (GradientDescentAccumulator.DEBUG) {
      GradientDescentAccumulator.log.debug(String.format("Input: %s & %s", weightGradient, Arrays.toString(data)));
    }
    final DoubleMatrix matrixA = new DoubleMatrix(data.length, 1, data);
    final DoubleMatrix matrixB = new DoubleMatrix(weightGradient.getDims()[0], weightGradient.getDims()[1], weightGradient.getData());
    final double[] inverted = matrixB.mmul(matrixA).data;
    if (GradientDescentAccumulator.DEBUG) {
      GradientDescentAccumulator.log.debug(String.format("Processing feedback inversion to produce deltas: %s", Arrays.toString(inverted)));
    }
    assert inverted.length == length();
    this.sink.feed(inverted);
  }

  public int length() {
    return this.sink.length();
  }

}
