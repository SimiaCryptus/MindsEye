package com.simiacryptus.mindseye.learning;

import org.jblas.DoubleMatrix;

import com.simiacryptus.mindseye.NDArray;

public class DeltaInversionBuffer {

  private int bufferPos = 0;
  private NDArray gradientBuffer;

  private final double minInversionRatio;
  private double[] signalBuffer;
  private final DeltaBuffer sink;

  
  protected DeltaInversionBuffer() {
    super();
    minInversionRatio = 0;
    sink = null;
  }

  public DeltaInversionBuffer(final double minInversionRatio, final DeltaBuffer sink) {
    this.sink = sink;
    this.minInversionRatio = minInversionRatio;
  }

  public void feed(final NDArray weightGradient, final double[] data) {
    if (0 == this.bufferPos) {
      final int inx = length();
      final int outx = data.length;
      int endx = data.length;
      while (endx < this.minInversionRatio * inx) {
        endx += outx;
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
          new DoubleMatrix(gradient.getDims()[0], dims[1], gradient.data).transpose(),
          new DoubleMatrix(this.signalBuffer.length, 1, this.signalBuffer)).data;
      assert inverted.length == length();
      this.sink.feed(inverted);
      this.bufferPos = 0;
    }
  }

  public int length() {
    return this.sink.length();
  }

}
