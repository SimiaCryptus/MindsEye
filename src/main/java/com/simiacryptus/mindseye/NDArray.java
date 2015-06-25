package com.simiacryptus.mindseye;

import java.util.Arrays;
import java.util.Iterator;
import java.util.function.ToDoubleBiFunction;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.codehaus.groovy.classgen.ReturnAdder;

public class NDArray {
  
  public final double[] data;
  private final int[] skips;
  private final int[] dims;
  
  public NDArray(int... dims) {
    this(dims, new double[dim(dims)]);
  }
  
  public NDArray(int[] dims, double[] data) {
    this.dims = Arrays.copyOf(dims, dims.length);
    this.skips = new int[dims.length];
    for (int i = 0; i < this.skips.length; i++)
    {
      if (i == 0)
        skips[i] = 1;
      else {
        skips[i] = skips[i - 1] * dims[i - 1];
      }
    }
    assert (dim(dims) == data.length);
    assert (0 < data.length);
    this.data = data;// Arrays.copyOf(data, data.length);
  }
  
  public static int dim(int... dims) {
    int total = 1;
    for (int i = 0; i < dims.length; i++)
    {
      total *= dims[i];
    }
    return total;
  }
  
  public interface UnivariateFunction {
    double apply(double v);
  }
  
  public NDArray map(ToDoubleBiFunction<Double, Coordinate> f) {
    return new NDArray(dims, coordStream().mapToDouble(i -> f.applyAsDouble(get(i), i)).toArray());
  }
  
  public NDArray map(UnivariateFunction f) {
    Arrays.parallelSetAll(data, i -> {
      double x = data[i];
      assert (Double.isFinite(x));
      double v = f.apply(x);
      assert (Double.isFinite(v));
      return v;
    });
    return this;
  }
  
  public double get(Coordinate coords) {
    double v = data[coords.index];
    assert (Double.isFinite(v));
    return v;
  }
  
  public double get(int... coords) {
    double v = data[index(coords)];
    assert (Double.isFinite(v));
    return v;
  }
  
  public void set(int[] coords, double value) {
    assert (Double.isFinite(value));
    set(index(coords), value);
  }
  
  public void set(Coordinate coords, double value) {
    assert (Double.isFinite(value));
    set(coords.index, value);
  }
  
  public void set(int index, double value) {
    assert (Double.isFinite(value));
    data[index] = value;
  }
  
  public int index(Coordinate coords) {
    return coords.index;
  }
  
  public int index(int... coords) {
    int v = 0;
    for (int i = 0; i < skips.length; i++) {
      v += skips[i] * coords[i];
    }
    return v;
    // return IntStream.range(0, skips.length).map(i->skips[i]*coords[i]).sum();
  }
  
  public int[] getDims() {
    return dims;
  }
  
  public double[] getData() {
    return data;
  }
  
  public int dim() {
    return data.length;
  }
  
  public void add(Coordinate coords, double value) {
    add(coords.index, value);
  }
  
  public void add(int[] coords, double value) {
    add(index(coords), value);
  }
  
  public void add(int index, double value) {
    assert (Double.isFinite(value));
    data[index] += value;
  }
  
  public Stream<Coordinate> coordStream() {
    return Util.toStream(new Iterator<Coordinate>() {
      
      int[] val = new int[dims.length];
      int cnt;
      
      @Override
      public boolean hasNext() {
        return cnt < dim();
      }
      
      @Override
      public Coordinate next() {
        int[] last = Arrays.copyOf(val, val.length);
        for (int i = 0; i < dims.length; i++)
        {
          if (++val[i] >= dims[i])
            val[i] = 0;
          else break;
        }
        assert (index(last) == cnt);
        return new Coordinate(cnt++, last);
      }
    }, dim());
  }
  
  public double sum() {
    double v = 0;
    for (int i = 0; i < data.length; i++)
    {
      v += data[i];
    }
    assert (Double.isFinite(v));
    return v;
  }
  
}
