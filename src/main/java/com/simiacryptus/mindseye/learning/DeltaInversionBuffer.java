package com.simiacryptus.mindseye.learning;

import org.jblas.DoubleMatrix;

import com.simiacryptus.mindseye.NDArray;

public class DeltaInversionBuffer {
  
  private final double minInversionRatio;
  private final DeltaBuffer sink;
  
  private NDArray gradientBuffer;
  private double[] signalBuffer;
  private int bufferPos = 0;
  
  public DeltaInversionBuffer(double minInversionRatio, DeltaBuffer sink) {
    this.sink = sink;
    this.minInversionRatio = minInversionRatio;
  }
  
  public int length() {
    return sink.length();
  }
  
  public void feed(NDArray weightGradient, double[] data) {
    if (0 == bufferPos) {
      int inx = length();
      int outx = data.length;
      int endx = data.length;
      while (endx < minInversionRatio * inx)
        endx += outx;
      gradientBuffer = new NDArray(inx, endx);
      signalBuffer = new double[endx];
    }
    for (int i = 0; i < data.length; i++)
    {
      for (int j = 0; j < length(); j++)
      {
        gradientBuffer.set(new int[] { j, bufferPos }, weightGradient.get(j, i));
      }
      signalBuffer[bufferPos] = data[i];
      bufferPos++;
    }
    if (bufferPos >= gradientBuffer.getDims()[1]) {
      final NDArray gradient = gradientBuffer;
      int[] dims = gradient.getDims();
      double[] inverted = org.jblas.Solve.solveLeastSquares(
          new DoubleMatrix(gradient.getDims()[0], dims[1], gradient.data).transpose(),
          new DoubleMatrix(signalBuffer.length, 1, signalBuffer)).data;
      assert (inverted.length == length());
      sink.feed(inverted);
      bufferPos = 0;
    }
  }
  
}
