package com.simiacryptus.mindseye;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleBiFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.jblas.DoubleMatrix;

public class NDArray {

  public static int dim(final int... dims) {
    int total = 1;
    for (final int dim : dims) {
      total *= dim;
    }
    return total;
  }

  private volatile double[] data;
  protected final int[] dims;
  protected final int[] skips;

  protected NDArray() {
    super();
    this.data = null;
    this.skips = null;
    this.dims = null;
  }

  public NDArray(final int... dims) {
    this(dims, null);
  }

  public NDArray(final int[] dims, final double[] data) {
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

  private int[] _add(final int[] base, final int... extra) {
    final int[] copy = Arrays.copyOf(base, base.length + extra.length);
    for (int i = 0; i < extra.length; i++) {
      copy[i + base.length] = extra[i];
    }
    return copy;
  }

  public void add(final Coordinate coords, final double value) {
    add(coords.index, value);
  }

  public final NDArray add(final int index, final double value) {
    getData()[index] += value;
    return this;
  }

  public void add(final int[] coords, final double value) {
    add(index(coords), value);
  }

  public NDArray add(final NDArray r) {
    final NDArray result = new NDArray(getDims());
    final double[] resultData = result.getData();
    final double[] rdata = r.getData();
    final double[] data = getData();
    for (int i = 0; i < data.length; i++) {
      resultData[i] = data[i] + rdata[i];
    }
    return result;
  }

  public DoubleMatrix asRowMatrix() {
    return new DoubleMatrix(this.dims[0], 1, getData()).transpose();
  }

  public Stream<Coordinate> coordStream() {
    return coordStream(false);
  }

  public Stream<Coordinate> coordStream(final boolean paralell) {
    return Util.toStream(new Iterator<Coordinate>() {

      int cnt = 0;
      int[] val = new int[NDArray.this.dims.length];

      @Override
      public boolean hasNext() {
        return this.cnt < dim();
      }

      @Override
      public Coordinate next() {
        final int[] last = Arrays.copyOf(this.val, this.val.length);
        for (int i = 0; i < this.val.length; i++) {
          if (++this.val[i] >= NDArray.this.dims[i]) {
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

  public NDArray copy() {
    return new NDArray(Arrays.copyOf(this.dims, this.dims.length), Arrays.copyOf(getData(), getData().length));
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
    final NDArray other = (NDArray) obj;
    if (!Arrays.equals(getData(), other.getData()))
      return false;
    if (!Arrays.equals(this.dims, other.dims))
      return false;
    return true;
  }

  public double get(final Coordinate coords) {
    final double v = getData()[coords.index];
    assert Double.isFinite(v);
    return v;
  }

  public double get(final int... coords) {
    // assert
    // IntStream.range(dims.length,coords.length).allMatch(i->coords[i]==0);
    // assert coords.length==dims.length;
    final double v = getData()[index(coords)];
    //assert Double.isFinite(v);
    return v;
  }

  public final double[] getData() {
    if (null == this.data) {
      synchronized (this) {
        if (null == this.data) {
          this.data = new double[NDArray.dim(this.dims)];
        }
      }
    }
    return this.data;
  }

  public final int[] getDims() {
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

  public NDArray map(final ToDoubleBiFunction<Double, Coordinate> f) {
    return new NDArray(this.dims, coordStream(false).mapToDouble(i -> f.applyAsDouble(get(i), i)).toArray());
  }

  public NDArray map(final java.util.function.DoubleUnaryOperator f) {
    final double[] cpy = new double[getData().length];
    for (int i = 0; i < getData().length; i++) {
      final double x = getData()[i];
      //assert Double.isFinite(x);
      final double v = f.applyAsDouble(x);
      //assert Double.isFinite(v);
      cpy[i] = v;
    }
    ;
    return new NDArray(this.dims, cpy);
  }

  public double rms(final NDArray right) {
    double sum = 0;
    for (int i = 0; i < this.dim(); i++) {
      final double diff = getData()[i] - right.getData()[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum / this.dim());
  }

  public NDArray scale(final double d) {
    for (int i = 0; i < getData().length; i++) {
      getData()[i] *= d;
    }
    return this;
  }

  public void set(final Coordinate coords, final double value) {
    assert Double.isFinite(value);
    set(coords.index, value);
  }

  public NDArray set(final double[] data) {
    for (int i = 0; i < getData().length; i++) {
      getData()[i] = data[i];
    }
    return this;
  }

  public NDArray set(final int index, final double value) {
    //assert Double.isFinite(value);
    getData()[index] = value;
    return this;
  }

  public void set(final int[] coords, final double value) {
    assert Double.isFinite(value);
    set(index(coords), value);
  }

  public double sum() {
    double v = 0;
    for (final double element : getData()) {
      v += element;
    }
    //assert Double.isFinite(v);
    return v;
  }

  @Override
  public String toString() {
    return toString(new int[] {});
  }

  private String toString(final int... coords) {
    if (coords.length == this.dims.length)
      return Double.toString(get(coords));
    else {
      List<String> list = IntStream.range(0, this.dims[coords.length]).mapToObj(i -> {
        return toString(_add(coords, i));
      }).collect(Collectors.<String>toList());
      if (list.size() > 10) {
        list = list.subList(0, 8);
        list.add("...");
      }
      final Optional<String> str = list.stream().limit(10).reduce((a, b) -> a + "," + b);
      return "[ " + str.get() + " ]";
    }
  }

  public NDArray reformat(int... dims) {
    return new NDArray(dims, getData());
  }

  public NDArray fill(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.getData(), i -> f.getAsDouble());
    return this;
  }


  public NDArray minus(NDArray right) {
    assert(Arrays.equals(getDims(), right.getDims()));
    NDArray copy = new NDArray(getDims());
    double[] thisData = getData();
    double[] rightData = right.getData();
    Arrays.parallelSetAll(copy.getData(), i -> thisData[i] - rightData[i]);
    return copy;
  }

  public void set(NDArray right) {
    assert(dim()==right.dim());
    double[] rightData = right.getData();
    Arrays.parallelSetAll(getData(), i -> rightData[i]);
  }

  public double l2() {
    return Math.sqrt(Arrays.stream(getData()).map(x->x*x).sum());
  }

  public double l1() {
    return Arrays.stream(getData()).sum();
  }



}
