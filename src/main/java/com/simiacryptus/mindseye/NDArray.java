package com.simiacryptus.mindseye;

import java.util.Arrays;
import java.util.Iterator;
import java.util.function.ToDoubleBiFunction;
import java.util.stream.Stream;

public class NDArray {
  
  public interface UnivariateFunction {
    double apply(double v);
  }
  
  public static int dim(final int... dims) {
    int total = 1;
    for (final int dim : dims) {
      total *= dim;
    }
    return total;
  }
  
  public final double[] data;
  
  private final int[] dims;
  
  private final int[] skips;
  
  protected NDArray() {
    super();
    data = null;
    skips = null;
    dims = null;
  }
  
  public NDArray(final int... dims) {
    this(dims, new double[NDArray.dim(dims)]);
  }
  
  public NDArray(final int[] dims, final double[] data) {
    this.dims = Arrays.copyOf(dims, dims.length);
    this.skips = new int[dims.length];
    for (int i = 0; i < this.skips.length; i++)
    {
      if (i == 0) {
        this.skips[i] = 1;
      } else {
        this.skips[i] = this.skips[i - 1] * dims[i - 1];
      }
    }
    assert NDArray.dim(dims) == data.length;
    assert 0 < data.length;
    this.data = data;// Arrays.copyOf(data, data.length);
  }
  
  public void add(final Coordinate coords, final double value) {
    add(coords.index, value);
  }
  
  public void add(final int index, final double value) {
    assert Double.isFinite(value);
    this.data[index] += value;
  }
  
  public void add(final int[] coords, final double value) {
    add(index(coords), value);
  }
  
  public Stream<Coordinate> coordStream() {
    return Util.toStream(new Iterator<Coordinate>() {
      
      int cnt;
      int[] val = new int[NDArray.this.dims.length];
      
      @Override
      public boolean hasNext() {
        return this.cnt < dim();
      }
      
      @Override
      public Coordinate next() {
        final int[] last = Arrays.copyOf(this.val, this.val.length);
        for (int i = 0; i < NDArray.this.dims.length; i++)
        {
          if (++this.val[i] >= NDArray.this.dims[i]) {
            this.val[i] = 0;
          } else {
            break;
          }
        }
        assert index(last) == this.cnt;
        return new Coordinate(this.cnt++, last);
      }
    }, dim());
  }
  
  public int dim() {
    return this.data.length;
  }
  
  public double get(final Coordinate coords) {
    final double v = this.data[coords.index];
    assert Double.isFinite(v);
    return v;
  }
  
  public double get(final int... coords) {
    final double v = this.data[index(coords)];
    assert Double.isFinite(v);
    return v;
  }
  
  public double[] getData() {
    return this.data;
  }
  
  public int[] getDims() {
    return this.dims;
  }
  
  public int index(final Coordinate coords) {
    return coords.index;
  }
  
  public int index(final int... coords) {
    int v = 0;
    for (int i = 0; i < this.skips.length; i++) {
      v += this.skips[i] * coords[i];
    }
    return v;
    // return IntStream.range(0, skips.length).map(i->skips[i]*coords[i]).sum();
  }
  
  public NDArray map(final ToDoubleBiFunction<Double, Coordinate> f) {
    return new NDArray(this.dims, coordStream().mapToDouble(i -> f.applyAsDouble(get(i), i)).toArray());
  }
  
  public NDArray map(final UnivariateFunction f) {
    Arrays.parallelSetAll(this.data, i -> {
      final double x = this.data[i];
      assert Double.isFinite(x);
      final double v = f.apply(x);
      assert Double.isFinite(v);
      return v;
    });
    return this;
  }
  
  public void scale(final double d) {
    for (int i = 0; i < this.data.length; i++)
    {
      this.data[i] *= d;
    }
  }
  
  public void set(final Coordinate coords, final double value) {
    assert Double.isFinite(value);
    set(coords.index, value);
  }
  
  public void set(final int index, final double value) {
    assert Double.isFinite(value);
    this.data[index] = value;
  }
  
  public void set(final int[] coords, final double value) {
    assert Double.isFinite(value);
    set(index(coords), value);
  }
  
  public double sum() {
    double v = 0;
    for (final double element : this.data) {
      v += element;
    }
    assert Double.isFinite(v);
    return v;
  }
  
}
