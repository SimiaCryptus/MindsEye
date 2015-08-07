package com.simiacryptus.mindseye.learning;

import java.util.Arrays;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;

public class DeltaInversionBuffer {
  public static boolean DEBUG = false;
  private static final Logger log = LoggerFactory.getLogger(DeltaInversionBuffer.class);
  
  private int bufferPos = 0;
  private NDArray gradientBuffer;
  
  private final double minInversionRatio;
  private double[] signalBuffer;
  private final DeltaSink sink;
  
  protected DeltaInversionBuffer() {
    super();
    this.minInversionRatio = 0;
    this.sink = null;
  }
  
  public DeltaInversionBuffer(final double minInversionRatio, final DeltaSink sink) {
    this.sink = sink;
    this.minInversionRatio = minInversionRatio;
  }
  
  public synchronized void feed(final NDArray weightGradient, final double[] data) {
    if(null == weightGradient) throw new IllegalArgumentException();
    if(null == data) throw new IllegalArgumentException();
    if (DeltaInversionBuffer.DEBUG) {
      DeltaInversionBuffer.log.debug(String.format("Input: %s & %s", weightGradient, Arrays.toString(data)));
    }
    if (0 == this.bufferPos) {
      final int inx = length();
      final int outx = data.length;
      int endx = data.length;
      while (endx < this.minInversionRatio * inx) {
        endx += outx;
      }
      if (DeltaInversionBuffer.DEBUG) {
        DeltaInversionBuffer.log.debug(String.format("Initialized with %s rows for %s weights and %s signal values", endx, inx, outx));
      }
      this.gradientBuffer = new NDArray(inx, endx);
      this.signalBuffer = new double[endx];
    }
    for (int i = 0; i < data.length; i++)
    {
      for (int j = 0; j < length(); j++)
      {
        this.gradientBuffer.set(new int[] { j, this.bufferPos }, weightGradient.get(j, i));
      }
      this.signalBuffer[this.bufferPos] = data[i];
      this.bufferPos++;
    }
    if (this.bufferPos >= this.gradientBuffer.getDims()[1]) {
      final NDArray gradient = this.gradientBuffer;
      final int[] dims = gradient.getDims();
      final double[] inverted = org.jblas.Solve.solveLeastSquares(
          new DoubleMatrix(gradient.getDims()[0], dims[1], gradient.getData()).transpose(),
          new DoubleMatrix(this.signalBuffer.length, 1, this.signalBuffer)).data;
      if (DeltaInversionBuffer.DEBUG) {
        DeltaInversionBuffer.log.debug(String.format("Processing feedback inversion to produce deltas: %s", Arrays.toString(inverted)));
      }
      assert inverted.length == length();
      this.sink.feed(inverted);
      this.bufferPos = 0;
    }
  }
  
  public int length() {
    return this.sink.length();
  }
  
}
