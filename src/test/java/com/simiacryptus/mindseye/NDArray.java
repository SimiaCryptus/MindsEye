package com.simiacryptus.mindseye;

import java.util.Arrays;
import java.util.stream.IntStream;

public class NDArray {

  final double[] data;
  private final int[] skips;
  private final int[] dims;

  public NDArray(int... dims) {
    this.dims = Arrays.copyOf(dims, dims.length);
    this.skips = new int[dims.length];
    for(int i=0;i<this.skips.length;i++)
    {
      if(i==0) skips[i] = 1;
      else {
        skips[i] = skips[i-1] * dims[i-1];
      }
    }
    this.data = new double[dim(dims)];
  }

  public NDArray(int[] dims, double[] data) {
    this(dims);
    Arrays.parallelSetAll(this.data, i->data[i]);
  }

  public static int dim(int... dims) {
    return IntStream.of(dims).reduce((a,b)->a*b).getAsInt();
  }
  
  public double get(int... coords) {
    return data[index(coords)];
  }
  
  public void set(int[] coords, double value) {
    int index = index(coords);
    set(index, value);
  }

  public void set(int index, double value) {
    data[index] = value;
  }

  public int index(int... coords) {
    return IntStream.range(0, skips.length).map(i->skips[i]*coords[i]).sum();
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

  public void add(int[] coords, double value) {
    add(index(coords), value);
  }

  public void add(int index, double value) {
    data[index] += value;
  }
  
}
