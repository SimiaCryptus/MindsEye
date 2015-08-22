package com.simiacryptus.mindseye;

import java.util.stream.DoubleStream;

public class LogNDArray extends NDArrayDouble {

  public LogNDArray() {
    super();
  }

  public LogNDArray(int... dims) {
    super(dims);
  }

  public LogNDArray(int[] dims, double[] data) {
    this(new NDArray(dims, data));
  }

  public LogNDArray(NDArray ndArray) {
    super(ndArray.dims, log(ndArray.data));
  }

  private static double[] log(double[] data) {
    return DoubleStream.of(data).map(x->Math.log(x)).toArray();
  }

  public NDArray exp() {
    return new NDArray(getDims(), DoubleStream.of(getData()).map(x->Math.log(x)).toArray());
  }

  public synchronized void add(final int index, final double value) {
    assert Double.isFinite(value);
    set(index, add(getData()[index], value));
  }

  public void add(final int[] coords, final double value) {
    add(index(coords), value);
  }

  private double add(double a, double b) {
    assert(Double.isFinite(a));
    assert(Double.isFinite(b));
    double r = Math.log(Math.exp(a) + Math.exp(b));
    assert(Double.isFinite(r));
    return r;
  }

  public LogNDArray scale(double rate) {
    double log = Math.log(rate);
    for(int i=0;i<data.length;i++) {
      data[i] += log;
    }
    return this;
  }

}