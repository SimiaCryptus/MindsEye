package com.simiacryptus.mindseye;

import java.util.Arrays;
import java.util.function.ToDoubleBiFunction;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;

public class NDArray extends NDArrayDouble {
  
  public NDArray() {
    super();
  }

  public NDArray(int... dims) {
    super(dims);
  }

  public NDArray(int[] dims, double[] data) {
    super(dims, data);
  }

  public LogNDArray log() {
    return new LogNDArray(this);
  }
  
  public void add(final Coordinate coords, final double value) {
    add(coords.index, value);
  }

  public synchronized void add(final int index, final double value) {
    assert Double.isFinite(value);
    getData()[index] += value;
  }

  public void add(final int[] coords, final double value) {
    add(index(coords), value);
  }

  public NDArray copy() {
    return new NDArray(Arrays.copyOf(this.dims, this.dims.length), Arrays.copyOf(getData(), getData().length));
  }

  public NDArray map(final ToDoubleBiFunction<Double, Coordinate> f) {
    return new NDArray(this.dims, coordStream(false).mapToDouble(i -> f.applyAsDouble(get(i), i)).toArray());
  }

  public NDArray map(final UnivariateFunction f) {
    final double[] cpy = new double[getData().length];
    for (int i = 0; i < getData().length; i++) {
      final double x = getData()[i];
      assert Double.isFinite(x);
      final double v = f.apply(x);
      assert Double.isFinite(v);
      cpy[i] = v;
    }
    ;
    return new NDArray(this.dims, cpy);
  }
  
  public NDArray scale(final double d) {
    for (int i = 0; i < getData().length; i++)
    {
      getData()[i] *= d;
    }
    return this;
  }
  
  public double sum() {
    double v = 0;
    for (final double element : getData()) {
      v += element;
    }
    assert Double.isFinite(v);
    return v;
  }
  
}
