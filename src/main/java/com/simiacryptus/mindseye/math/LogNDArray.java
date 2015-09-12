package com.simiacryptus.mindseye.math;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import com.simiacryptus.mindseye.util.Util;

public class LogNDArray {

  public static int dim(final int... dims) {
    int total = 1;
    for (final int dim : dims) {
      total *= dim;
    }
    return total;
  }

  private static LogNumber[] log(final double[] data) {
    return DoubleStream.of(data).mapToObj(x -> LogNumber.log(x)).toArray(i -> new LogNumber[i]);
  }

  protected volatile LogNumber[] data;
  protected final int[] dims;

  protected final int[] skips;

  protected LogNDArray() {
    super();
    this.data = null;
    this.skips = null;
    this.dims = null;
  }

  public LogNDArray(final int... dims) {
    this(dims, null);
  }

  public LogNDArray(final int[] dims, final LogNumber[] data) {
    this.dims = Arrays.copyOf(dims, dims.length);
    this.skips = new int[dims.length];
    for (int i = 0; i < this.skips.length; i++) {
      if (i == 0) {
        this.skips[i] = 1;
      } else {
        this.skips[i] = this.skips[i - 1] * dims[i - 1];
      }
    }
    assert null == data || NDArray.dim(dims) == data.length;
    assert null == data || 0 < data.length;
    this.data = data;// Arrays.copyOf(data, data.length);
  }

  public LogNDArray(final LogNDArray copy) {
    this(copy.dims, Arrays.copyOf(copy.data, copy.data.length));
  }

  public LogNDArray(final NDArray ndArray) {
    this(ndArray.dims, LogNDArray.log(ndArray.data));
  }

  public synchronized void add(final int index, final double value) {
    assert Double.isFinite(value);
    set(index, getData()[index].add(value));
  }

  public synchronized void add(final int index, final LogNumber value) {
    assert value.isFinite();
    set(index, getData()[index].add(value));
  }

  public void add(final int[] coords, final LogNumber value) {
    add(index(coords), value);
  }

  private int[] concat(final int[] a, final int... b) {
    final int[] copy = Arrays.copyOf(a, a.length + b.length);
    for (int i = 0; i < b.length; i++) {
      copy[i + a.length] = b[i];
    }
    return copy;
  }

  public Stream<Coordinate> coordStream() {
    return coordStream(false);
  }

  public Stream<Coordinate> coordStream(final boolean paralell) {
    return Util.toStream(new Iterator<Coordinate>() {

      int cnt = 0;
      int[] val = new int[LogNDArray.this.dims.length];

      @Override
      public boolean hasNext() {
        return this.cnt < dim();
      }

      @Override
      public Coordinate next() {
        final int[] last = Arrays.copyOf(this.val, this.val.length);
        for (int i = 0; i < this.val.length; i++) {
          if (++this.val[i] >= LogNDArray.this.dims[i]) {
            this.val[i] = 0;
          } else {
            break;
          }
        }
        final int index = this.cnt++;
        // assert index(last) == index;
        return new Coordinate(index, last);
      }
    }, dim(), paralell);
  }

  public int dim() {
    return getData().length;
  }

  @Override
  public boolean equals(final Object obj) {
    if (this == obj)
      return true;
    if (obj == null)
      return false;
    if (getClass() != obj.getClass())
      return false;
    final LogNDArray other = (LogNDArray) obj;
    if (!Arrays.equals(getData(), other.getData()))
      return false;
    if (!Arrays.equals(this.dims, other.dims))
      return false;
    return true;
  }

  public NDArray exp() {
    return new NDArray(getDims(), Stream.of(getData()).mapToDouble(x -> x.doubleValue()).toArray());
  }

  public LogNumber get(final Coordinate coords) {
    final LogNumber v = getData()[coords.index];
    assert Double.isFinite(v.logValue);
    return v;
  }

  public LogNumber get(final int... coords) {
    // assert
    // IntStream.range(dims.length,coords.length).allMatch(i->coords[i]==0);
    // assert coords.length==dims.length;
    final LogNumber v = getData()[index(coords)];
    assert v.isFinite();
    return v;
  }

  public LogNumber[] getData() {
    if (null == this.data) {
      synchronized (this) {
        if (null == this.data) {
          this.data = new LogNumber[NDArray.dim(this.dims)];
          Arrays.fill(this.data, LogNumber.ZERO);
        }
      }
    }
    return this.data;
  }

  public int[] getDims() {
    return this.dims;
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + Arrays.hashCode(getData());
    result = prime * result + Arrays.hashCode(this.dims);
    return result;
  }

  public int index(final Coordinate coords) {
    return coords.index;
  }

  public int index(final int... coords) {
    int v = 0;
    for (int i = 0; i < this.skips.length && i < coords.length; i++) {
      v += this.skips[i] * coords[i];
    }
    return v;
    // return IntStream.range(0, skips.length).map(i->skips[i]*coords[i]).sum();
  }

  public LogNDArray scale(final double rate) {
    final LogNDArray copy = new LogNDArray(this);
    final LogNumber log = LogNumber.log(rate);
    for (int i = 0; i < this.data.length; i++) {
      copy.set(i, this.data[i].multiply(log));
    }
    return copy;
  }

  public void set(final Coordinate coords, final LogNumber value) {
    assert value.isFinite();
    set(coords.index, value);
  }

  public void set(final int index, final LogNumber value) {
    assert value.isFinite();
    getData()[index] = value;
  }

  public void set(final int[] coords, final LogNumber value) {
    assert value.isFinite();
    set(index(coords), value);
  }

  public LogNDArray set(final LogNumber[] data) {
    for (int i = 0; i < getData().length; i++) {
      getData()[i] = data[i];
    }
    return this;
  }

  @Override
  public String toString() {
    return toString(new int[] {});
  }

  private String toString(final int... coords) {
    if (coords.length == this.dims.length)
      return get(coords).toString();
    else {
      List<String> list = IntStream.range(0, this.dims[coords.length]).mapToObj(i -> {
        return toString(concat(coords, i));
      }).collect(Collectors.<String>toList());
      if (list.size() > 10) {
        list = list.subList(0, 8);
        list.add("...");
      }
      final Optional<String> str = list.stream().limit(10).reduce((a, b) -> a + "," + b);
      return "{ " + str.get() + " }";
    }
  }

  public LogNDArray add(LogNDArray right) {
    LogNDArray sum = new LogNDArray(getDims());
    for(int i=0;i<dim();i++) {
      LogNumber[] thisData = getData();
      LogNumber[] rdata = right.getData();
      sum.add(i, rdata[i].add(thisData[i]));
    }
    return sum;
  }

}
