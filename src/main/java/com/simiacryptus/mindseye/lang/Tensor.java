/*
 * Copyright (c) 2017 by Andrew Charneski.
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

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.simiacryptus.util.io.JsonUtil;

import java.awt.image.BufferedImage;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.*;
import java.util.stream.*;

/**
 * The type Tensor.
 */
public class Tensor implements Serializable {
  
  private static ConcurrentHashMap<Integer, BlockingQueue<double[]>> recycling = new ConcurrentHashMap<>();
  
  static {
    java.util.concurrent.Executors.newScheduledThreadPool(1).scheduleAtFixedRate(new Runnable() {
      @Override
      public void run() {
        recycling.forEach((k, v) -> {
          v.clear();
        });
      }
    }, 0, 10, TimeUnit.SECONDS);
  }
  
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
    this.data = null;
    this.strides = null;
    this.dimensions = null;
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
   * Instantiates a new Tensor.
   *
   * @param data the data
   * @param dims the dims
   */
  public Tensor(final double[] data, final int... dims) {
    this.dimensions = Arrays.copyOf(dims, dims.length);
    this.strides = getSkips(dims);
    //this.data = data;// Arrays.copyOf(data, data.length);
    if (null != data) {
      this.data = obtain(data.length);// Arrays.copyOf(data, data.length);
      System.arraycopy(data, 0, this.data, 0, data.length);
    }
    //assert (null == data || Tensor.dim(dims) == data.length);
  }
  
  /**
   * Instantiates a new Tensor.
   *
   * @param data the data
   * @param dims the dims
   */
  public Tensor(final float[] data, final int... dims) {
    this.dimensions = Arrays.copyOf(dims, dims.length);
    this.strides = getSkips(dims);
    if (null != data) {
      this.data = obtain(data.length);// Arrays.copyOf(data, data.length);
      Arrays.parallelSetAll(this.data, i -> {
        double v = data[i];
        return Double.isFinite(v) ? v : 0;
      });
      assert Arrays.stream(this.data).allMatch(v -> Double.isFinite(v));
    }
    //assert (null == data || Tensor.dim(dims) == data.length);
  }
  
  private static int[] getSkips(int[] dims) {
    int[] skips = new int[dims.length];
    for (int i = 0; i < skips.length; i++) {
      if (i == 0) {
        skips[i] = 1;
      }
      else {
        skips[i] = skips[i - 1] * dims[i - 1];
      }
    }
    return skips;
  }
  
  /**
   * Instantiates a new Tensor.
   *
   * @param ds the ds
   */
  public Tensor(double[] ds) {
    this(ds, new int[]{ds.length});
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
   * Recycle.
   *
   * @param data the data
   */
  public static void recycle(double[] data) {
    if (null == data) return;
    //if(null != data) return;
    if (data.length < 256) return;
    BlockingQueue<double[]> bin = recycling.get(data.length);
    if (null == bin) {
      //System.err.println("New Recycle Bin: " + data.length);
      bin = new ArrayBlockingQueue<double[]>(Math.max(1,(int) (1e8 / data.length)));
      recycling.put(data.length, bin);
    }
    bin.offer(data);
  }
  
  /**
   * Obtain double [ ].
   *
   * @param length the length
   * @return the double [ ]
   */
  public static double[] obtain(int length) {
    BlockingQueue<double[]> bin = recycling.get(length);
    if (null != bin) {
      double[] data = bin.poll();
      if (null != data) {
        Arrays.fill(data, 0);
        return data;
      }
    }
    try {
      return new double[length];
    } catch (java.lang.OutOfMemoryError e) {
      try {
        clear();
        System.gc();
        return new double[length];
      } catch (java.lang.OutOfMemoryError e2) {
        throw new com.simiacryptus.mindseye.lang.OutOfMemoryError("Could not allocate " + length + " bytes", e2);
      }
    }
  }
  
  /**
   * Clear.
   */
  public static void clear() {
    recycling.clear();
  }
  
  /**
   * From rgb tensor.
   *
   * @param img the img
   * @return the tensor
   */
  public static Tensor fromRGB(BufferedImage img) {
    int width = img.getWidth();
    int height = img.getHeight();
    Tensor a = new Tensor(width, height, 3);
    int[] coords = {0, 0, 0};
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        coords[0] = x;
        coords[1] = y;
        coords[2] = 0;
        a.set(coords, img.getRGB(x, y) & 0xFF);
        coords[2] = 1;
        a.set(coords, img.getRGB(x, y) >> 8 & 0xFF);
        coords[2] = 2;
        a.set(coords, img.getRGB(x, y) >> 16 & 0x0FF);
      }
    }
    return a;
  }
  
  private static int bound8bit(final int value) {
    final int max = 0xFF;
    final int min = 0;
    return (value < min) ? min : ((value > max) ? max : value);
  }
  
  private static double bound8bit(final double value) {
    final int max = 0xFF;
    final int min = 0;
    return (value < min) ? min : ((value > max) ? max : value);
  }
  
  @Override
  public void finalize() throws Throwable {
    release();
    super.finalize();
  }
  
  private byte references = 1;
  
  /**
   * Acquire reference short.
   *
   * @return the short
   */
  public short acquireReference() {
    return ++references;
  }
  
  /**
   * Release boolean.
   *
   * @return the boolean
   */
  public boolean release() {
    if(--references <= 0) {
      if (null != data) {
        recycle(data);
        data = null;
        return true;
      }
    }
    return false;
  }
  
  private int[] _add(final int[] base, final int... extra) {
    final int[] copy = Arrays.copyOf(base, base.length + extra.length);
    for (int i = 0; i < extra.length; i++) {
      copy[i + base.length] = extra[i];
    }
    return copy;
  }
  
  /**
   * Add.
   *
   * @param coords the coords
   * @param value  the value
   */
  public void add(final Coordinate coords, final double value) {
    add(coords.index, value);
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
   * Coord stream stream.
   *
   * @return the stream
   */
  public Stream<Coordinate> coordStream() {
    return StreamSupport.stream(Spliterators.spliterator(new Iterator<Coordinate>() {
      
      int cnt = 0;
      int[] val = new int[Tensor.this.dimensions.length];
      
      @Override
      public boolean hasNext() {
        return this.cnt < dim();
      }
      
      @Override
      public synchronized Coordinate next() {
        if (0 < this.cnt) {
          for (int i = 0; i < this.val.length; i++) {
            if (++this.val[i] >= Tensor.this.dimensions[i]) {
              this.val[i] = 0;
            }
            else {
              break;
            }
          }
        }
        // assert index(last) == index;
        return new Coordinate(this.cnt++, this.val);
      }
    }, dim(), Spliterator.ORDERED), false);
  }
  
  /**
   * Copy tensor.
   *
   * @return the tensor
   */
  public Tensor copy() {
    return new Tensor(Arrays.copyOf(getData(), getData().length), Arrays.copyOf(this.dimensions, this.dimensions.length));
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
    if (!Arrays.equals(this.dimensions, other.dimensions)) {
      return false;
    }
    return true;
  }
  
  /**
   * Fill tensor.
   *
   * @param f the f
   * @return the tensor
   */
  public Tensor fill(final DoubleSupplier f) {
    Arrays.parallelSetAll(getData(), i -> f.getAsDouble());
    return this;
  }
  
  /**
   * Fill tensor.
   *
   * @param f the f
   * @return the tensor
   */
  public Tensor fill(final java.util.function.IntToDoubleFunction f) {
    Arrays.parallelSetAll(getData(), i -> f.applyAsDouble(i));
    return this;
  }
  
  /**
   * Get double.
   *
   * @param coords the coords
   * @return the double
   */
  public double get(final Coordinate coords) {
    final double v = getData()[coords.index];
    return v;
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
   * @param c1     the c 1
   * @param c2     the c 2
   * @param c3     the c 3
   * @param coords the coords
   * @return the double
   */
  public double get(final int c1, final int c2, final int c3, final int... coords) {
    return getData()[index(c1, c2, c3, coords)];
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
    if (null == this.data) {
      synchronized (this) {
        if (null == this.data) {
          int length = Tensor.dim(this.dimensions);
          this.data = obtain(length);
        }
      }
    }
    assert (dim() == this.data.length);
    return this.data;
  }
  
  /**
   * Get dimensions int [ ].
   *
   * @return the int [ ]
   */
  public final int[] getDimensions() {
    return this.dimensions;
  }
  
  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + Arrays.hashCode(getData());
    result = prime * result + Arrays.hashCode(this.dimensions);
    return result;
  }
  
  /**
   * Index int.
   *
   * @param coords the coords
   * @return the int
   */
  public int index(final Coordinate coords) {
    return coords.index;
  }
  
  /**
   * Index int.
   *
   * @param coords the coords
   * @return the int
   */
  public int index(final int[] coords) {
    int v = 0;
    for (int i = 0; i < this.strides.length && i < coords.length; i++) {
      v += this.strides[i] * coords[i];
    }
    return v;
    // return IntStream.range(0, strides.length).mapCoords(i->strides[i]*coords[i]).sum();
  }
  
  /**
   * Index int.
   *
   * @param c1 the c 1
   * @return the int
   */
  public int index(final int c1) {
    int v = 0;
    v += this.strides[0] * c1;
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
    v += this.strides[0] * c1;
    v += this.strides[1] * c2;
    return v;
    // return IntStream.range(0, strides.length).mapCoords(i->strides[i]*coords[i]).sum();
  }
  
  /**
   * Index int.
   *
   * @param c1     the c 1
   * @param c2     the c 2
   * @param c3     the c 3
   * @param coords the coords
   * @return the int
   */
  public int index(final int c1, final int c2, final int c3, final int... coords) {
    int v = 0;
    v += this.strides[0] * c1;
    v += this.strides[1] * c2;
    v += this.strides[2] * c3;
    if(null != coords && 0 < coords.length) for (int i = 0; (3+i) < this.strides.length && i < coords.length; i++) {
      v += this.strides[3+i] * coords[3+i];
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
    final double[] cpy = new double[getData().length];
    for (int i = 0; i < getData().length; i++) {
      final double x = getData()[i];
      // assert Double.isFinite(x);
      final double v = f.applyAsDouble(x);
      // assert Double.isFinite(v);
      cpy[i] = v;
    }
    ;
    return new Tensor(cpy, this.dimensions);
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
  
  /**
   * Map tensor.
   *
   * @param f the f
   * @return the tensor
   */
  public Tensor mapIndex(final TupleOperator f) {
    return new Tensor(getDoubles(IntStream.range(0,dim()).mapToDouble(i -> f.eval(get(i), i)), dim()), this.dimensions);
  }
  
  /**
   * Map tensor.
   *
   * @param f the f
   * @return the tensor
   */
  public Tensor mapCoords(final ToDoubleBiFunction<Double, Coordinate> f) {
    return new Tensor(getDoubles(coordStream().mapToDouble(i -> f.applyAsDouble(get(i), i)), dim()), this.dimensions);
  }
  
  /**
   * Map parallel tensor.
   *
   * @param right the right
   * @param f     the f
   * @return the tensor
   */
  public Tensor reduceParallel(Tensor right, final DoubleBinaryOperator f) {
    double[] dataL = getData();
    double[] dataR = right.getData();
    return new Tensor(getDoubles(IntStream.range(0, dim()).mapToDouble(i->f.applyAsDouble(dataL[i],dataR[i])), dim()), this.dimensions);
  }
  
  /**
   * Map parallel tensor.
   *
   * @param f the f
   * @return the tensor
   */
  public Tensor mapParallel(final DoubleUnaryOperator f) {
    double[] data = getData();
    return new Tensor(getDoubles(IntStream.range(0, dim()).mapToDouble(i->f.applyAsDouble(data[i])), dim()), this.dimensions);
  }
  
  /**
   * Get doubles double [ ].
   *
   * @param stream the stream
   * @param dim    the dim
   * @return the double [ ]
   */
  public static double[] getDoubles(DoubleStream stream, int dim) {
    final double[] doubles = obtain(dim);
    AtomicInteger j = new AtomicInteger();
    stream.forEach(v->doubles[j.getAndIncrement()] = v);
    return doubles;
  }
  
  /**
   * Sets parallel.
   *
   * @param f the f
   */
  public Tensor fillByCoord(final ToDoubleFunction<Coordinate> f) {
    coordStream().forEach(c -> set(c, f.applyAsDouble(c)));
    return this;
  }
  
  /**
   * Sets parallel.
   *
   * @param f the f
   */
  public void setParallelByIndex(final IntToDoubleFunction f) {
    IntStream.range(0, dim()).forEach(c -> set(c, f.applyAsDouble(c)));
  }
  
  /**
   * Minus tensor.
   *
   * @param right the right
   * @return the tensor
   */
  public Tensor minus(final Tensor right) {
    assert Arrays.equals(getDimensions(), right.getDimensions());
    final Tensor copy = new Tensor(getDimensions());
    final double[] thisData = getData();
    final double[] rightData = right.getData();
    Arrays.parallelSetAll(copy.getData(), i -> thisData[i] - rightData[i]);
    return copy;
  }
  
  /**
   * Reformat tensor.
   *
   * @param dims the dims
   * @return the tensor
   */
  public Tensor reformat(final int... dims) {
    return new Tensor(getData(), dims);
  }
  
  /**
   * Multiply tensor.
   *
   * @param d the d
   * @return the tensor
   */
  public Tensor multiply(final double d) {
    Tensor tensor = new Tensor(getDimensions());
    double[] resultData = tensor.getData();
    double[] thisData = getData();
    for (int i = 0; i < thisData.length; i++) {
      resultData[i] = d * thisData[i];
    }
    return tensor;
  }
  
  /**
   * Scale tensor (in-place)
   *
   * @param d the d
   * @return the tensor
   */
  public Tensor scale(final double d) {
    double[] data = getData();
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
    set(coords.index, value);
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
   * To image buffered image.
   *
   * @return the buffered image
   */
  public BufferedImage toImage() {
    int[] dims = getDimensions();
    if (3 == dims.length) {
      if (3 == dims[2]) {
        return toRgbImage();
      }
      else {
        assert (1 == dims[2]);
        return toGrayImage();
      }
    }
    else {
      assert (2 == dims.length);
      return toGrayImage();
    }
  }
  
  public List<BufferedImage> toImages() {
    int[] dims = getDimensions();
    if (3 == dims.length) {
      if (3 == dims[2]) {
        return Arrays.asList(toRgbImage());
      }
      else if (0 == dims[2] % 3) {
        ArrayList<BufferedImage> list = new ArrayList<>();
        for(int i=0;i<dims[2];i+=3) {
          list.add(toRgbImage(i,i+1,i+2));
        }
        return list;
      }
      else if (1 == dims[2]) {
        return Arrays.asList(toGrayImage());
      }
      else {
        ArrayList<BufferedImage> list = new ArrayList<>();
        for(int i=0;i<dims[2];i++) {
          list.add(toGrayImage(i));
        }
        return list;
      }
    }
    else {
      assert (2 == dims.length);
      return Arrays.asList(toGrayImage());
    }
  }
  
  /**
   * To gray image buffered image.
   *
   * @return the buffered image
   */
  public BufferedImage toGrayImage() {
    return toGrayImage(0);
  }
  
  public BufferedImage toGrayImage(int band) {
    int width = getDimensions()[0];
    int height = getDimensions()[1];
    BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
    for (int x = 0; x < width; x++)
      for (int y = 0; y < height; y++) {
        double v = get(x, y, band);
        image.getRaster().setSample(x, y, 0, v < 0 ? 0 : v > 255 ? 255 : v);
      }
    return image;
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
   * @return the buffered image
   * @param redBand
   * @param greenBand
   * @param blueBand
   */
  public BufferedImage toRgbImage(int redBand, int greenBand, int blueBand) {
    final int[] dims = this.getDimensions();
    final BufferedImage img = new BufferedImage(dims[0], dims[1], BufferedImage.TYPE_INT_RGB);
    for (int x = 0; x < img.getWidth(); x++) {
      for (int y = 0; y < img.getHeight(); y++) {
        if (this.getDimensions()[2] == 1) {
          final double value = this.get(x, y, 0);
          img.setRGB(x, y, bound8bit((int) value) * 0x010101);
        }
        else {
          final double red = bound8bit(this.get(x, y, redBand));
          final double green = bound8bit(this.get(x, y, greenBand));
          final double blue = bound8bit(this.get(x, y, blueBand));
          img.setRGB(x, y, (int) (red + ((int) green << 8) + ((int) blue << 16)));
        }
      }
    }
    return img;
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
   * @param coords the coords
   * @param value  the value
   */
  public void set(final int[] coords, final double value) {
    assert Double.isFinite(value);
    set(index(coords), value);
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
  public double mean() { return sum() / dim();}
  public double rms() { return Math.sqrt(sumSq() / dim());}
  
  public double sumSq() {
    double v = 0;
    for (final double element : getData()) {
      v += element*element;
    }
    // assert Double.isFinite(v);
    return v;
  }
  
  @Override
  public String toString() {
    return toString(new int[]{});
  }
  
  private String toString(final int... coords) {
    if (coords.length == this.dimensions.length) {
      return Double.toString(get(coords));
    }
    else {
      List<String> list = IntStream.range(0, this.dimensions[coords.length]).mapToObj(i -> {
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
  
  /**
   * Sets all.
   *
   * @param v the v
   */
  public void setAll(double v) {
    double[] data = getData();
    for (int i = 0; i < data.length; i++) {
      data[i] = v;
    }
  }
  
  /**
   * Size int.
   *
   * @return the int
   */
  public int size() {
    return null == data ? Tensor.dim(this.dimensions) : data.length;
  }
  
  /**
   * Gets json.
   *
   * @return the json
   */
  public JsonElement getJson() {
    JsonObject json = new JsonObject();
    json.add("dimensions", JsonUtil.getJson(dimensions));
    if (data != null) json.add("data", JsonUtil.getJson(data));
    return json;
  }
  
  /**
   * From json tensor.
   *
   * @param json the json
   * @return the tensor
   */
  public static Tensor fromJson(JsonObject json) {
    if (null == json) return null;
    int[] dims = JsonUtil.getIntArray(json.getAsJsonArray("dimensions"));
    double[] data = json.has("data") ? JsonUtil.getDoubleArray(json.getAsJsonArray("data")) : null;
    return new Tensor(data, dims);
  }
  
  /**
   * Add tensor.
   *
   * @param left  the left
   * @param right the right
   * @return the tensor
   */
  public static Tensor add(Tensor left, Tensor right) {
    assert Arrays.equals(left.getDimensions(), right.getDimensions());
    Tensor result = new Tensor(left.getDimensions());
    double[] resultData = result.getData();
    double[] leftData = left.getData();
    double[] rightData = right.getData();
    for (int i = 0; i < resultData.length; i++) {
      double l = leftData[i];
      double r = rightData[i];
      resultData[i] = l + r;
    }
    return result;
  }
  
  /**
   * Product tensor.
   *
   * @param left  the left
   * @param right the right
   * @return the tensor
   */
  public static Tensor product(Tensor left, Tensor right) {
    if(left.dim() == 1 && right.dim() != 1) return product(right, left);
    assert left.dim() == right.dim() || 1 == right.dim();
    Tensor result = new Tensor(left.getDimensions());
    double[] resultData = result.getData();
    double[] leftData = left.getData();
    double[] rightData = right.getData();
    for (int i = 0; i < resultData.length; i++) {
      double l = leftData[i];
      double r = rightData[1==rightData.length?rightData.length-1:i];
      resultData[i] = l * r;
    }
    return result;
  }
  
  /**
   * Get data as floats float [ ].
   *
   * @return the float [ ]
   */
  public float[] getDataAsFloats() {
    return toFloats(getData());
  }
  
  /**
   * To floats float [ ].
   *
   * @param data the data
   * @return the float [ ]
   */
  public static float[] toFloats(double[] data) {
    float[] buffer = new float[data.length];
    for (int i = 0; i < data.length; i++) {
      buffer[i] = (float) data[i];
    }
    return buffer;
  }
  
  /**
   * To doubles double [ ].
   *
   * @param data the data
   * @return the double [ ]
   */
  public static double[] toDoubles(float[] data) {
    double[] buffer = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      buffer[i] = data[i];
    }
    return buffer;
  }
  
  /**
   * Add tensor.
   *
   * @param tensor the tensor
   * @return the tensor
   */
  public Tensor add(Tensor tensor) {
    assert (Arrays.equals(getDimensions(), tensor.getDimensions()));
    final ToDoubleBiFunction<Double, Coordinate> f = (v, c) -> v + tensor.get(c);
    return mapCoords(f);
  }
  
  /**
   * Accum.
   *
   * @param tensor the tensor
   */
  public void accum(Tensor tensor) {
    assert (Arrays.equals(getDimensions(), tensor.getDimensions()));
    setParallelByIndex(c -> get(c) + tensor.get(c));
  }
}
