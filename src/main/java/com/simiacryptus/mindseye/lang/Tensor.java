/*
 * Copyright (c) 2018 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.lang;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import org.apache.commons.lang3.ArrayUtils;

import java.awt.image.BufferedImage;
import java.io.Serializable;
import java.util.*;
import java.util.function.*;
import java.util.stream.*;

/**
 * A multi-dimensional array of data. Represented internally as a single double[] array. This class is central to data
 * handling in MindsEye, and may have some odd-looking or suprising optimizations.
 */
@SuppressWarnings("serial")
public class Tensor implements Serializable {
  
  /**
   * The constant json_precision.
   */
  public static DataSerializer json_precision = SerialPrecision.Float;
  /**
   * The Dimensions.
   */
  protected final int[] dimensions;
  /**
   * The Strides.
   */
  protected final int[] strides;
  /**
   * The Data.
   */
  protected volatile double[] data;
  
  /**
   * Instantiates a new Tensor.
   */
  protected Tensor() {
    super();
    data = null;
    strides = null;
    dimensions = null;
  }
  
  /**
   * Instantiates a new Tensor.
   *
   * @param ds the ds
   */
  public Tensor(final double... ds) {
    this(ds, ds.length);
  }
  
  /**
   * Instantiates a new Tensor.
   *
   * @param data the data
   * @param dims the dims
   */
  public Tensor(final double[] data, final int... dims) {
    dimensions = Arrays.copyOf(dims, dims.length);
    strides = Tensor.getSkips(dims);
    //this.data = data;// Arrays.copyOf(data, data.length);
    if (null != data) {
      this.data = RecycleBin.DOUBLES.obtain(data.length);// Arrays.copyOf(data, data.length);
      System.arraycopy(data, 0, this.data, 0, data.length);
    }
    assert isValid();
    //assert (null == data || Tensor.dim(dims) == data.length);
  }
  
  private Tensor(int[] dimensions, int[] strides, double[] data) {
    assert null == data || data.length == Tensor.dim(dimensions);
    this.dimensions = dimensions;
    this.strides = strides;
    this.data = data;
    assert isValid();
  }
  
  /**
   * Instantiates a new Tensor.
   *
   * @param data the data
   * @param dims the dims
   */
  public Tensor(final float[] data, final int... dims) {
    dimensions = Arrays.copyOf(dims, dims.length);
    strides = Tensor.getSkips(dims);
    if (null != data) {
      this.data = RecycleBin.DOUBLES.obtain(data.length);// Arrays.copyOf(data, data.length);
      Arrays.parallelSetAll(this.data, i -> {
        final double v = data[i];
        return Double.isFinite(v) ? v : 0;
      });
      assert Arrays.stream(this.data).allMatch(v -> Double.isFinite(v));
    }
    assert isValid();
    //assert (null == data || Tensor.dim(dims) == data.length);
  }
  
  /**
   * From json tensor.
   *
   * @param json      the json
   * @param resources the resources
   * @return the tensor
   */
  public static Tensor fromJson(final JsonElement json, Map<String, byte[]> resources) {
    if (null == json) return null;
    if (json.isJsonArray()) {
      final JsonArray array = json.getAsJsonArray();
      final int size = array.size();
      if (array.get(0).isJsonPrimitive()) {
        final double[] doubles = IntStream.range(0, size).mapToObj(i -> {
          return array.get(i);
        }).mapToDouble(element -> {
          return element.getAsDouble();
        }).toArray();
        Tensor tensor = new Tensor(doubles);
        assert tensor.isValid();
        return tensor;
      }
      else {
        final List<Tensor> elements = IntStream.range(0, size).mapToObj(i -> {
          return array.get(i);
        }).map(element -> {
          return Tensor.fromJson(element, resources);
        }).collect(Collectors.toList());
        final int[] dimensions = elements.get(0).getDimensions();
        if (!elements.stream().allMatch(t -> Arrays.equals(dimensions, t.getDimensions()))) {
          throw new IllegalArgumentException();
        }
        final int[] newDdimensions = Arrays.copyOf(dimensions, dimensions.length + 1);
        newDdimensions[dimensions.length] = size;
        final Tensor tensor = new Tensor(newDdimensions);
        final double[] data = tensor.getData();
        for (int i = 0; i < size; i++) {
          final double[] e = elements.get(i).getData();
          System.arraycopy(e, 0, data, i * e.length, e.length);
        }
        assert tensor.isValid();
        return tensor;
      }
    }
    else if (json.isJsonObject()) {
      JsonObject jsonObject = json.getAsJsonObject();
      int[] dims = fromJsonArray(jsonObject.getAsJsonArray("dim"));
      Tensor tensor = new Tensor(dims);
      SerialPrecision precision = SerialPrecision.valueOf(jsonObject.getAsJsonPrimitive("precision").getAsString());
      JsonElement base64 = jsonObject.get("base64");
      if (null == base64) {
        if (null == resources) throw new IllegalArgumentException("No Data Resources");
        String resourceId = jsonObject.getAsJsonPrimitive("resource").getAsString();
        tensor.setBytes(resources.get(resourceId), precision);
      }
      else {
        tensor.setBytes(Base64.getDecoder().decode(base64.getAsString()), precision);
      }
      assert tensor.isValid();
      return tensor;
    }
    else {
      Tensor tensor = new Tensor(json.getAsJsonPrimitive().getAsDouble());
      assert tensor.isValid();
      return tensor;
    }
  }
  
  /**
   * Instantiates a new Tensor.
   *
   * @param dims the dims
   */
  public Tensor(final int... dims) {
    this((double[]) null, dims);
  }
  
  /**
   * Add tensor.
   *
   * @param left  the left
   * @param right the right
   * @return the tensor
   */
  public static Tensor add(final Tensor left, final Tensor right) {
    if (left.dim() == 1 && right.dim() != 1) return Tensor.add(right, left);
    assert Arrays.equals(left.getDimensions(), right.getDimensions());
    final Tensor result = new Tensor(left.getDimensions());
    final double[] resultData = result.getData();
    final double[] leftData = left.getData();
    final double[] rightData = right.getData();
    for (int i = 0; i < resultData.length; i++) {
      final double l = leftData[i];
      final double r = rightData[1 == rightData.length ? 0 : i];
      resultData[i] = l + r;
    }
    return result;
  }
  
  private static double bound8bit(final double value) {
    final int max = 0xFF;
    final int min = 0;
    return value < min ? min : value > max ? max : value;
  }
  
  private static int bound8bit(final int value) {
    final int max = 0xFF;
    final int min = 0;
    return value < min ? min : value > max ? max : value;
  }
  
  /**
   * Dim int.
   *
   * @param dims the dims
   * @return the int
   */
  public static int dim(final int... dims) {
    int total = 1;
    for (final int dim : dims) {
      total *= dim;
    }
    return total;
  }
  
  /**
   * Reorder dimensions tensor.
   *
   * @param tensor the tensor
   * @param fn     the fn
   * @return the tensor
   */
  public static Tensor reorderDimensions(Tensor tensor, UnaryOperator<int[]> fn) {
    Tensor result = new Tensor(fn.apply(tensor.getDimensions()));
    tensor.coordStream(false).forEach(c -> {
      result.set(fn.apply(c.getCoords()), tensor.get(c));
    });
    return result;
  }
  
  /**
   * From rgb tensor.
   *
   * @param img the img
   * @return the tensor
   */
  public static Tensor fromRGB(final BufferedImage img) {
    final int width = img.getWidth();
    final int height = img.getHeight();
    final Tensor a = new Tensor(width, height, 3);
    IntStream.range(0, width).parallel().forEach(x -> {
      final int[] coords = {0, 0, 0};
      IntStream.range(0, height).forEach(y -> {
        coords[0] = x;
        coords[1] = y;
        coords[2] = 0;
        a.set(coords, img.getRGB(x, y) & 0xFF);
        coords[2] = 1;
        a.set(coords, img.getRGB(x, y) >> 8 & 0xFF);
        coords[2] = 2;
        a.set(coords, img.getRGB(x, y) >> 16 & 0x0FF);
      });
    });
    return a;
  }
  
  /**
   * Get doubles double [ ].
   *
   * @param stream the stream
   * @param dim    the dim
   * @return the double [ ]
   */
  public static double[] getDoubles(final DoubleStream stream, final int dim) {
    final double[] doubles = RecycleBin.DOUBLES.obtain(dim);
    stream.forEach(new DoubleConsumer() {
      int j = 0;
      
      @Override
      public void accept(final double value) {
        doubles[j++] = value;
      }
    });
    return doubles;
  }
  
  private static int[] getSkips(final int[] dims) {
    final int[] skips = new int[dims.length];
    for (int i = 0; i < skips.length; i++) {
      if (i == 0) {
        skips[0] = 1;
      }
      else {
        skips[i] = skips[i - 1] * dims[i - 1];
      }
    }
    return skips;
  }
  
  /**
   * Product tensor.
   *
   * @param left  the left
   * @param right the right
   * @return the tensor
   */
  public static Tensor product(final Tensor left, final Tensor right) {
    if (left.dim() == 1 && right.dim() != 1) return Tensor.product(right, left);
    assert left.dim() == right.dim() || 1 == right.dim();
    final Tensor result = new Tensor(left.getDimensions());
    final double[] resultData = result.getData();
    final double[] leftData = left.getData();
    final double[] rightData = right.getData();
    for (int i = 0; i < resultData.length; i++) {
      final double l = leftData[i];
      final double r = rightData[1 == rightData.length ? 0 : i];
      resultData[i] = l * r;
    }
    return result;
  }
  
  /**
   * To doubles double [ ].
   *
   * @param data the data
   * @return the double [ ]
   */
  public static double[] toDoubles(final float[] data) {
    final double[] buffer = RecycleBin.DOUBLES.obtain(data.length);
    for (int i = 0; i < data.length; i++) {
      buffer[i] = data[i];
    }
    return buffer;
  }
  
  /**
   * To floats float [ ].
   *
   * @param data the data
   * @return the float [ ]
   */
  public static float[] toFloats(final double[] data) {
    final float[] buffer = new float[data.length];
    for (int i = 0; i < data.length; i++) {
      buffer[i] = (float) data[i];
    }
    return buffer;
  }
  
  /**
   * To json array json array.
   *
   * @param ints the ints
   * @return the json array
   */
  public static JsonArray toJsonArray(int[] ints) {
    JsonArray dim = new JsonArray();
    for (int i = 0; i < ints.length; i++) {
      dim.add(new JsonPrimitive(ints[i]));
    }
    return dim;
  }
  
  /**
   * From json array int [ ].
   *
   * @param ints the ints
   * @return the int [ ]
   */
  public static int[] fromJsonArray(JsonArray ints) {
    int[] array = new int[ints.size()];
    for (int i = 0; i < ints.size(); i++) {
      array[i] = ints.get(i).getAsInt();
    }
    return array;
  }
  
  /**
   * Reverse dimensions tensor.
   *
   * @param tensor the tensor
   * @return the tensor
   */
  public static Tensor reverseDimensions(Tensor tensor) {
    return reorderDimensions(tensor, Tensor::reverse);
  }
  
  /**
   * Permute dimensions tensor.
   *
   * @param tensor the tensor
   * @param key    the key
   * @return the tensor
   */
  public static Tensor permuteDimensions(Tensor tensor, int... key) {
    return reorderDimensions(tensor, in -> permute(key, in));
  }
  
  /**
   * Is valid boolean.
   *
   * @return the boolean
   */
  public boolean isValid() {
    return null == this.data || this.data.length == Tensor.dim(dimensions);
  }
  
  /**
   * Permute int [ ].
   *
   * @param key  the key
   * @param data the data
   * @return the int [ ]
   */
  public static int[] permute(int[] key, int[] data) {
    int[] copy = new int[key.length];
    for (int i = 0; i < key.length; i++) {
      copy[i] = data[key[i]];
    }
    return copy;
  }
  
  /**
   * Reverse int [ ].
   *
   * @param dimensions the dimensions
   * @return the int [ ]
   */
  public static int[] reverse(int[] dimensions) {
    int[] copy = Arrays.copyOf(dimensions, dimensions.length);
    ArrayUtils.reverse(copy);
    return copy;
  }
  
  /**
   * Accum.
   *
   * @param tensor the tensor
   */
  public void accumulate(final Tensor tensor) {
    assert Arrays.equals(getDimensions(), tensor.getDimensions());
    setParallelByIndex(c -> get(c) + tensor.get(c));
  }
  
  /**
   * Add.
   *
   * @param coords the coords
   * @param value  the value
   */
  public void add(final Coordinate coords, final double value) {
    add(coords.getIndex(), value);
  }
  
  /**
   * Add tensor.
   *
   * @param index the index
   * @param value the value
   * @return the tensor
   */
  public final Tensor add(final int index, final double value) {
    getData()[index] += value;
    return this;
  }
  
  /**
   * Add.
   *
   * @param coords the coords
   * @param value  the value
   */
  public void add(final int[] coords, final double value) {
    add(index(coords), value);
  }
  
  /**
   * Add right.
   *
   * @param right the right
   * @return the right
   */
  public Tensor add(final Tensor right) {
    assert Arrays.equals(getDimensions(), right.getDimensions());
    return mapCoords((c) -> get(c) + right.get(c));
  }
  
  /**
   * Coord stream stream.
   *
   * @param parallel the safe
   * @return the stream
   */
  public Stream<Coordinate> coordStream(boolean parallel) {
    //ConcurrentHashSet<Object> distinctBuffer = new ConcurrentHashSet<>();
    //assert distinctBuffer.add(coordinate.copy()) : String.format("Duplicate: %s in %s", coordinate, distinctBuffer);
    return StreamSupport.stream(Spliterators.spliterator(new Iterator<Coordinate>() {
      
      int cnt = 0;
      Coordinate coordinate = new Coordinate();
      int[] val = new int[dimensions.length];
      int[] safeCopy = new int[dimensions.length];
      
      @Override
      public boolean hasNext() {
        return cnt < dim();
      }
      
      @Override
      public synchronized Coordinate next() {
        if (0 < cnt) {
          for (int i = 0; i < val.length; i++) {
            if (++val[i] >= dimensions[i]) {
              val[i] = 0;
            }
            else {
              break;
            }
          }
        }
        System.arraycopy(val, 0, safeCopy, 0, val.length);
        coordinate.setIndex(cnt++);
        coordinate.setCoords(safeCopy);
        return parallel ? coordinate.copy() : coordinate;
      }
    }, dim(), Spliterator.ORDERED), parallel);
  }
  
  /**
   * Copy tensor.
   *
   * @return the tensor
   */
  public Tensor copy() {
    return new Tensor(Arrays.copyOf(getData(), getData().length), Arrays.copyOf(dimensions, dimensions.length));
  }
  
  /**
   * Dim int.
   *
   * @return the int
   */
  public int dim() {
    if (null != data) {
      return data.length;
    }
    else {
      return Tensor.dim(dimensions);
    }
  }
  
  @Override
  public boolean equals(final Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    final Tensor other = (Tensor) obj;
    if (!Arrays.equals(getData(), other.getData())) {
      return false;
    }
    return Arrays.equals(dimensions, other.dimensions);
  }
  
  @Override
  public void finalize() throws Throwable {
    if (null != data) {
      RecycleBin.DOUBLES.recycle(data, data.length);
      data = null;
    }
    super.finalize();
  }
  
  /**
   * Get double.
   *
   * @param coords the coords
   * @return the double
   */
  public double get(final Coordinate coords) {
    final double v = getData()[coords.getIndex()];
    return v;
  }
  
  /**
   * Get.
   *
   * @param bufferArray the buffer array
   */
  public void get(final double[] bufferArray) {
    System.arraycopy(getData(), 0, bufferArray, 0, dim());
  }
  
  /**
   * Get double.
   *
   * @param index the index
   * @return the double
   */
  public double get(final int index) {
    return getData()[index];
  }
  
  /**
   * Get double.
   *
   * @param c1 the c 1
   * @param c2 the c 2
   * @return the double
   */
  public double get(final int c1, final int c2) {
    return getData()[index(c1, c2)];
  }
  
  /**
   * Get double.
   *
   * @param c1 the c 1
   * @param c2 the c 2
   * @param c3 the c 3
   * @return the double
   */
  public double get(final int c1, final int c2, final int c3) {
    return getData()[index(c1, c2, c3)];
  }
  
  /**
   * Get double.
   *
   * @param c1     the c 1
   * @param c2     the c 2
   * @param c3     the c 3
   * @param c4     the c 4
   * @param coords the coords
   * @return the double
   */
  public double get(final int c1, final int c2, final int c3, final int c4, final int... coords) {
    return getData()[index(c1, c2, c3, c4, coords)];
  }
  
  /**
   * Get double.
   *
   * @param coords the coords
   * @return the double
   */
  public double get(final int[] coords) {
    return getData()[index(coords)];
  }
  
  /**
   * Get data double [ ].
   *
   * @return the double [ ]
   */
  public double[] getData() {
    if (null == data) {
      synchronized (this) {
        if (null == data) {
          final int length = Tensor.dim(dimensions);
          data = RecycleBin.DOUBLES.obtain(length);
          assert null != data;
          assert length == data.length;
        }
      }
    }
    assert isValid();
    assert null != data;
    return data;
  }
  
  /**
   * Get data as floats float [ ].
   *
   * @return the float [ ]
   */
  public float[] getDataAsFloats() {
    return Tensor.toFloats(getData());
  }
  
  /**
   * Get dimensions int [ ].
   *
   * @return the int [ ]
   */
  public final int[] getDimensions() {
    return Arrays.copyOf(dimensions, dimensions.length);
  }
  
  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + Arrays.hashCode(getData());
    result = prime * result + Arrays.hashCode(dimensions);
    return result;
  }
  
  /**
   * Index int.
   *
   * @param coords the coords
   * @return the int
   */
  public int index(final Coordinate coords) {
    return coords.getIndex();
  }
  
  /**
   * Index int.
   *
   * @param c1 the c 1
   * @return the int
   */
  public int index(final int c1) {
    int v = 0;
    v += strides[0] * c1;
    return v;
    // return IntStream.range(0, strides.length).mapCoords(i->strides[i]*coords[i]).sum();
  }
  
  /**
   * Index int.
   *
   * @param c1 the c 1
   * @param c2 the c 2
   * @return the int
   */
  public int index(final int c1, final int c2) {
    int v = 0;
    v += strides[0] * c1;
    v += strides[1] * c2;
    return v;
    // return IntStream.range(0, strides.length).mapCoords(i->strides[i]*coords[i]).sum();
  }
  
  /**
   * Index int.
   *
   * @param c1 the c 1
   * @param c2 the c 2
   * @param c3 the c 3
   * @return the int
   */
  public int index(final int c1, final int c2, final int c3) {
    int v = 0;
    v += strides[0] * c1;
    v += strides[1] * c2;
    v += strides[2] * c3;
    return v;
    // return IntStream.range(0, strides.length).mapCoords(i->strides[i]*coords[i]).sum();
  }
  
  /**
   * Index int.
   *
   * @param c1     the c 1
   * @param c2     the c 2
   * @param c3     the c 3
   * @param c4     the c 4
   * @param coords the coords
   * @return the int
   */
  public int index(final int c1, final int c2, final int c3, final int c4, final int... coords) {
    int v = 0;
    v += strides[0] * c1;
    v += strides[1] * c2;
    v += strides[2] * c3;
    v += strides[3] * c4;
    if (null != coords && 0 < coords.length) {
      for (int i = 0; 4 + i < strides.length && i < coords.length; i++) {
        v += strides[4 + i] * coords[4 + i];
      }
    }
    return v;
    // return IntStream.range(0, strides.length).mapCoords(i->strides[i]*coords[i]).sum();
  }
  
  /**
   * Index int.
   *
   * @param coords the coords
   * @return the int
   */
  public int index(final int[] coords) {
    int v = 0;
    for (int i = 0; i < strides.length && i < coords.length; i++) {
      v += strides[i] * coords[i];
    }
    return v;
    // return IntStream.range(0, strides.length).mapCoords(i->strides[i]*coords[i]).sum();
  }
  
  /**
   * L 1 double.
   *
   * @return the double
   */
  public double l1() {
    return Arrays.stream(getData()).sum();
  }
  
  /**
   * L 2 double.
   *
   * @return the double
   */
  public double l2() {
    return Math.sqrt(Arrays.stream(getData()).map(x -> x * x).sum());
  }
  
  /**
   * Map tensor.
   *
   * @param f the f
   * @return the tensor
   */
  public Tensor map(final java.util.function.DoubleUnaryOperator f) {
    final double[] data = getData();
    final double[] cpy = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      final double x = data[i];
      // assert Double.isFinite(x);
      final double v = f.applyAsDouble(x);
      // assert Double.isFinite(v);
      cpy[i] = v;
    }
    return new Tensor(cpy, dimensions);
  }
  
  /**
   * Map coords tensor.
   *
   * @param f the f
   * @return the tensor
   */
  public Tensor mapCoords(final ToDoubleFunction<Coordinate> f) {return mapCoords(f, true);}
  
  /**
   * Map coords tensor.
   *
   * @param f        the f
   * @param parallel
   * @return the tensor
   */
  public Tensor mapCoords(final ToDoubleFunction<Coordinate> f, boolean parallel) {
    return new Tensor(Tensor.getDoubles(coordStream(parallel).mapToDouble(i -> f.applyAsDouble(i)), dim()), dimensions);
  }
  
  /**
   * Map index tensor.
   *
   * @param f the f
   * @return the tensor
   */
  public Tensor mapIndex(final TupleOperator f) {
    return new Tensor(Tensor.getDoubles(IntStream.range(0, dim()).mapToDouble(i -> f.eval(get(i), i)), dim()), dimensions);
  }
  
  /**
   * Map parallel tensor.
   *
   * @param f the f
   * @return the tensor
   */
  public Tensor mapParallel(final DoubleUnaryOperator f) {
    final double[] data = getData();
    return new Tensor(Tensor.getDoubles(IntStream.range(0, dim()).mapToDouble(i -> f.applyAsDouble(data[i])), dim()), dimensions);
  }
  
  /**
   * Mean double.
   *
   * @return the double
   */
  public double mean() {
    return sum() / dim();
  }
  
  /**
   * Minus tensor.
   *
   * @param right the right
   * @return the tensor
   */
  public Tensor minus(final Tensor right) {
    if (!Arrays.equals(getDimensions(), right.getDimensions())) {
      throw new IllegalArgumentException(Arrays.toString(getDimensions()) + " != " + Arrays.toString(right.getDimensions()));
    }
    final Tensor copy = new Tensor(getDimensions());
    final double[] thisData = getData();
    final double[] rightData = right.getData();
    Arrays.parallelSetAll(copy.getData(), i -> thisData[i] - rightData[i]);
    return copy;
  }
  
  /**
   * Multiply tensor.
   *
   * @param d the d
   * @return the tensor
   */
  public Tensor multiply(final double d) {
    final Tensor tensor = new Tensor(getDimensions());
    final double[] resultData = tensor.getData();
    final double[] thisData = getData();
    for (int i = 0; i < thisData.length; i++) {
      resultData[i] = d * thisData[i];
    }
    return tensor;
  }
  
  /**
   * Pretty printGroups string.
   *
   * @return the string
   */
  public String prettyPrint() {
    return toString(true);
  }
  
  /**
   * Reduce parallel tensor.
   *
   * @param right the right
   * @param f     the f
   * @return the tensor
   */
  public Tensor reduceParallel(final Tensor right, final DoubleBinaryOperator f) {
    if (!Arrays.equals(right.getDimensions(), getDimensions())) {
      throw new IllegalArgumentException(Arrays.toString(right.getDimensions()) + " != " + Arrays.toString(getDimensions()));
    }
    final double[] dataL = getData();
    final double[] dataR = right.getData();
    return new Tensor(Tensor.getDoubles(IntStream.range(0, dim()).mapToDouble(i -> f.applyAsDouble(dataL[i], dataR[i])), dim()), dimensions);
  }
  
  /**
   * Rms double.
   *
   * @return the double
   */
  public double rms() {
    return Math.sqrt(sumSq() / dim());
  }
  
  /**
   * Round tensor.
   *
   * @param precision the precision
   * @return the tensor
   */
  public Tensor round(final int precision) {
    if (precision > 8) return this;
    if (precision < 1) throw new IllegalArgumentException();
    return round(precision, 10);
  }
  
  /**
   * Round tensor.
   *
   * @param precision the precision
   * @param base      the base
   * @return the tensor
   */
  public Tensor round(final int precision, final int base) {
    return map(v -> {
      final double units = Math.pow(base, Math.ceil(Math.log(v) / Math.log(base)) - precision);
      return Math.round(v / units) * units;
    });
  }
  
  /**
   * Scale tensor.
   *
   * @param d the d
   * @return the tensor
   */
  public Tensor scale(final double d) {
    return map(v -> v * d);
  }
  
  /**
   * Scale tensor.
   *
   * @param d the d
   * @return the tensor
   */
  public Tensor scaleInPlace(final double d) {
    final double[] data = getData();
    for (int i = 0; i < data.length; i++) {
      data[i] *= d;
    }
    return this;
  }
  
  /**
   * Set.
   *
   * @param coords the coords
   * @param value  the value
   */
  public void set(final Coordinate coords, final double value) {
    assert Double.isFinite(value);
    set(coords.getIndex(), value);
  }
  
  /**
   * Set tensor.
   *
   * @param data the data
   * @return the tensor
   */
  public Tensor set(final double[] data) {
    for (int i = 0; i < getData().length; i++) {
      getData()[i] = data[i];
    }
    return this;
  }
  
  /**
   * Fill tensor.
   *
   * @param f the f
   * @return the tensor
   */
  public Tensor set(final DoubleSupplier f) {
    Arrays.parallelSetAll(getData(), i -> f.getAsDouble());
    return this;
  }
  
  /**
   * Set tensor.
   *
   * @param index the index
   * @param value the value
   * @return the tensor
   */
  public Tensor set(final int index, final double value) {
    // assert Double.isFinite(value);
    getData()[index] = value;
    return this;
  }
  
  /**
   * Set.
   *
   * @param coord1 the coord 1
   * @param coord2 the coord 2
   * @param value  the value
   */
  public void set(final int coord1, final int coord2, final double value) {
    assert Double.isFinite(value);
    set(index(coord1, coord2), value);
  }
  
  /**
   * Set.
   *
   * @param coord1 the coord 1
   * @param coord2 the coord 2
   * @param coord3 the coord 3
   * @param value  the value
   */
  public void set(final int coord1, final int coord2, final int coord3, final double value) {
    assert Double.isFinite(value);
    set(index(coord1, coord2, coord3), value);
  }
  
  /**
   * Set.
   *
   * @param coord1 the coord 1
   * @param coord2 the coord 2
   * @param coord3 the coord 3
   * @param coord4 the coord 4
   * @param value  the value
   */
  public void set(final int coord1, final int coord2, final int coord3, final int coord4, final double value) {
    assert Double.isFinite(value);
    set(index(coord1, coord2, coord3, coord4), value);
  }
  
  /**
   * Set.
   *
   * @param coords the coords
   * @param value  the value
   */
  public void set(final int[] coords, final double value) {
    assert Double.isFinite(value);
    set(index(coords), value);
  }
  
  /**
   * Set tensor.
   *
   * @param f the f
   * @return the tensor
   */
  public Tensor set(final IntToDoubleFunction f) {
    Arrays.parallelSetAll(getData(), f);
    return this;
  }
  
  /**
   * Set.
   *
   * @param right the right
   */
  public void set(final Tensor right) {
    assert dim() == right.dim();
    final double[] rightData = right.getData();
    Arrays.parallelSetAll(getData(), i -> rightData[i]);
  }
  
  /**
   * Sets all.
   *
   * @param v the v
   */
  public void setAll(final double v) {
    final double[] data = getData();
    for (int i = 0; i < data.length; i++) {
      data[i] = v;
    }
  }
  
  /**
   * Fill by coord tensor.
   *
   * @param f the f
   * @return the tensor
   */
  public Tensor setByCoord(final ToDoubleFunction<Coordinate> f) {return setByCoord(f, true);}
  
  /**
   * Fill by coord tensor.
   *
   * @param f        the f
   * @param parallel
   * @return the tensor
   */
  public Tensor setByCoord(final ToDoubleFunction<Coordinate> f, boolean parallel) {
    coordStream(parallel).forEach(c -> set(c, f.applyAsDouble(c)));
    return this;
  }
  
  /**
   * Sets parallel by index.
   *
   * @param f the f
   */
  public void setParallelByIndex(final IntToDoubleFunction f) {
    IntStream.range(0, dim()).forEach(c -> set(c, f.applyAsDouble(c)));
  }
  
  /**
   * Size int.
   *
   * @return the int
   */
  public int size() {
    return null == data ? Tensor.dim(dimensions) : data.length;
  }
  
  /**
   * Sum double.
   *
   * @return the double
   */
  public double sum() {
    double v = 0;
    for (final double element : getData()) {
      v += element;
    }
    // assert Double.isFinite(v);
    return v;
  }
  
  /**
   * Sum sq double.
   *
   * @return the double
   */
  public double sumSq() {
    double v = 0;
    for (final double element : getData()) {
      v += element * element;
    }
    // assert Double.isFinite(v);
    return v;
  }
  
  /**
   * To gray image buffered image.
   *
   * @return the buffered image
   */
  public BufferedImage toGrayImage() {
    return toGrayImage(0);
  }
  
  /**
   * To gray image buffered image.
   *
   * @param band the band
   * @return the buffered image
   */
  public BufferedImage toGrayImage(final int band) {
    final int width = getDimensions()[0];
    final int height = getDimensions()[1];
    final BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        final double v = get(x, y, band);
        image.getRaster().setSample(x, y, 0, v < 0 ? 0 : v > 255 ? 255 : v);
      }
    }
    return image;
  }
  
  /**
   * To image buffered image.
   *
   * @return the buffered image
   */
  public BufferedImage toImage() {
    final int[] dims = getDimensions();
    if (3 == dims.length) {
      if (3 == dims[2]) {
        return toRgbImage();
      }
      else {
        assert 1 == dims[2];
        return toGrayImage();
      }
    }
    else {
      assert 2 == dims.length;
      return toGrayImage();
    }
  }
  
  /**
   * To images list.
   *
   * @return the list
   */
  public List<BufferedImage> toImages() {
    final int[] dims = getDimensions();
    if (3 == dims.length) {
      if (3 == dims[2]) {
        return Arrays.asList(toRgbImage());
      }
      else if (0 == dims[2] % 3) {
        final ArrayList<BufferedImage> list = new ArrayList<>();
        for (int i = 0; i < dims[2]; i += 3) {
          list.add(toRgbImage(i, i + 1, i + 2));
        }
        return list;
      }
      else if (1 == dims[2]) {
        return Arrays.asList(toGrayImage());
      }
      else {
        final ArrayList<BufferedImage> list = new ArrayList<>();
        for (int i = 0; i < dims[2]; i++) {
          list.add(toGrayImage(i));
        }
        return list;
      }
    }
    else {
      assert 2 == dims.length : "order: " + dims.length;
      return Arrays.asList(toGrayImage());
    }
  }
  
  /**
   * To json json element.
   *
   * @param resources      the resources
   * @param dataSerializer the data serializer
   * @return the json element
   */
  public JsonElement toJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    if (dim() > 1024) {
      JsonObject obj = new JsonObject();
      int[] dimensions = getDimensions();
      obj.add("dim", toJsonArray(dimensions));
      byte[] bytes = getBytes(dataSerializer);
      obj.addProperty("precision", ((SerialPrecision) dataSerializer).name());
      if (null != resources) {
        String id = UUID.randomUUID().toString();
        obj.addProperty("resource", id);
        resources.put(id, bytes);
      }
      else {
        obj.addProperty("base64", Base64.getEncoder().encodeToString(bytes));
      }
      return obj;
    }
    else {
      return toJson(new int[]{});
    }
  }
  
  /**
   * Sets bytes.
   *
   * @param bytes the bytes
   * @return the bytes
   */
  public Tensor setBytes(byte[] bytes) {return setBytes(bytes, json_precision);}
  
  /**
   * Get bytes byte [ ].
   *
   * @param precision the precision
   * @return the byte [ ]
   */
  public byte[] getBytes(DataSerializer precision) {
    return precision.toBytes(getData());
  }
  
  /**
   * Sets bytes.
   *
   * @param bytes     the bytes
   * @param precision the precision
   * @return the bytes
   */
  public Tensor setBytes(byte[] bytes, DataSerializer precision) {
    precision.copy(bytes, getData());
    return this;
  }
  
  private JsonElement toJson(final int[] coords) {
    if (coords.length == dimensions.length) {
      final double d = get(coords);
      return new JsonPrimitive(d);
    }
    else {
      final JsonArray jsonArray = new JsonArray();
      IntStream.range(0, dimensions[dimensions.length - (coords.length + 1)]).mapToObj(i -> {
        final int[] newCoord = new int[coords.length + 1];
        System.arraycopy(coords, 0, newCoord, 1, coords.length);
        newCoord[0] = i;
        return toJson(newCoord);
      }).forEach(l -> jsonArray.add(l));
      return jsonArray;
    }
  }
  
  /**
   * To rgb image buffered image.
   *
   * @return the buffered image
   */
  public BufferedImage toRgbImage() {
    return toRgbImage(0, 1, 2);
  }
  
  /**
   * To rgb image buffered image.
   *
   * @param redBand   the red band
   * @param greenBand the green band
   * @param blueBand  the blue band
   * @return the buffered image
   */
  public BufferedImage toRgbImage(final int redBand, final int greenBand, final int blueBand) {
    final int[] dims = getDimensions();
    final BufferedImage img = new BufferedImage(dims[0], dims[1], BufferedImage.TYPE_INT_RGB);
    for (int x = 0; x < img.getWidth(); x++) {
      for (int y = 0; y < img.getHeight(); y++) {
        if (getDimensions()[2] == 1) {
          final double value = this.get(x, y, 0);
          img.setRGB(x, y, Tensor.bound8bit((int) value) * 0x010101);
        }
        else {
          final double red = Tensor.bound8bit(this.get(x, y, redBand));
          final double green = Tensor.bound8bit(this.get(x, y, greenBand));
          final double blue = Tensor.bound8bit(this.get(x, y, blueBand));
          img.setRGB(x, y, (int) (red + ((int) green << 8) + ((int) blue << 16)));
        }
      }
    }
    return img;
  }
  
  @Override
  public String toString() {
    return toString(false);
  }
  
  private String toString(final boolean prettyPrint, final int... coords) {
    if (coords.length == dimensions.length) {
      return Double.toString(get(coords));
    }
    else {
      List<String> list = IntStream.range(0, dimensions[coords.length]).mapToObj(i -> {
        final int[] newCoord = Arrays.copyOf(coords, coords.length + 1);
        newCoord[coords.length] = i;
        return toString(prettyPrint, newCoord);
      }).collect(Collectors.toList());
      if (list.size() > 10) {
        list = list.subList(0, 8);
        list.add("...");
      }
      if (prettyPrint) {
        if (coords.length < dimensions.length - 2) {
          final String str = list.stream().limit(10)
                                 .map(s -> "\t" + s.replaceAll("\n", "\n\t"))
                                 .reduce((a, b) -> a + ",\n" + b).orElse("");
          return "[\n" + str + "\n]";
        }
        else {
          final String str = list.stream().reduce((a, b) -> a + ", " + b).orElse("");
          return "[ " + str + " ]";
        }
      }
      else {
        final String str = list.stream().reduce((a, b) -> a + "," + b).orElse("");
        return "[ " + str + " ]";
      }
    }
  }
  
  /**
   * Reverse dimensions tensor.
   *
   * @return the tensor
   */
  public Tensor reverseDimensions() {
    return reverseDimensions(this);
  }
  
  /**
   * Permute dimensions tensor.
   *
   * @param key the key
   * @return the tensor
   */
  public Tensor permuteDimensions(int... key) {
    return permuteDimensions(this, key);
  }
  
  /**
   * Reorder dimensions tensor.
   *
   * @param fn the fn
   * @return the tensor
   */
  public Tensor reorderDimensions(UnaryOperator<int[]> fn) {
    return reorderDimensions(this, fn);
  }
  
  /**
   * Reshape cast tensor.
   *
   * @param dims the dims
   * @return the tensor
   */
  public Tensor reshapeCast(int... dims) {
    if (0 == dims.length) throw new IllegalArgumentException();
    if (dim(dims) != dim()) throw new IllegalArgumentException();
    return new Tensor(dims, Tensor.getSkips(dims), getData());
  }
  
  /**
   * For each.
   *
   * @param fn the fn
   */
  public void forEach(CoordOperator fn) {forEach(fn, true);}
  
  /**
   * For each.
   *
   * @param fn       the fn
   * @param parallel the parallel
   */
  public void forEach(CoordOperator fn, boolean parallel) {
    coordStream(parallel).forEach(c -> {
      fn.eval(get(c), c);
    });
  }
  
  /**
   * The interface Coord operator.
   */
  public interface CoordOperator {
    /**
     * Eval double.
     *
     * @param value the value
     * @param index the index
     * @return the double
     */
    void eval(double value, Coordinate index);
  }
  
  
  /**
   * The interface Tuple operator.
   */
  public interface TupleOperator {
    /**
     * Eval double.
     *
     * @param value the value
     * @param index the index
     * @return the double
     */
    double eval(double value, int index);
  }
}
