package com.simiacryptus.mindseye;

import java.util.Arrays;
import java.util.Iterator;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class NDArray {

  public static class Coords {
    public final int[] coords;
    public final int index;
    
    public Coords(int index, int[] coords) {
      super();
      this.index = index;
      this.coords = coords;
    }
    
    @Override
    public String toString() {
      return Arrays.toString(coords);
    }
    
    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + Arrays.hashCode(coords);
      return result;
    }
    
    @Override
    public boolean equals(Object obj) {
      if (this == obj) return true;
      if (obj == null) return false;
      if (getClass() != obj.getClass()) return false;
      Coords other = (Coords) obj;
      if (!Arrays.equals(coords, other.coords)) return false;
      return true;
    }
    
  }
  
  public final double[] data;
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

  public double get(Coords coords) {
    return data[coords.index];
  }

  public double get(int... coords) {
    return data[index(coords)];
  }
  
  public void set(int[] coords, double value) {
    set(index(coords), value);
  }

  public void set(Coords coords, double value) {
    set(coords.index, value);
  }

  public void set(int index, double value) {
    data[index] = value;
  }

  public int index(Coords coords) {
    return coords.index;
  }

  public int index(int... coords) {
    int v = 0;
    for(int i=0;i<skips.length;i++) {
      v += skips[i]*coords[i];
    }
    return v;
    //return IntStream.range(0, skips.length).map(i->skips[i]*coords[i]).sum();
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

  public void add(Coords coords, double value) {
    add(coords.index, value);
  }

  public void add(int[] coords, double value) {
    add(index(coords), value);
  }

  public void add(int index, double value) {
    data[index] += value;
  }

  public Stream<Coords> coordStream() {
    return Util.toStream(new Iterator<Coords>() {

      int[] val = new int[dims.length];
      int cnt;
      
      @Override
      public boolean hasNext() {
        return cnt < dim();
      }

      @Override
      public Coords next() {
        int[] last = Arrays.copyOf(val, val.length);
        for(int i=0;i<dims.length;i++)
        {
          if(++val[i] >= dims[i]) val[i] = 0;
          else break;
        }
        assert(index(last) == cnt);
        return new Coords(cnt++, last);
      }
    }, dim());
  }
  
}
