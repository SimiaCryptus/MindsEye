package com.simiacryptus.mindseye.deltas;

import java.util.Arrays;
import java.util.Comparator;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.math.LogNumber;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.TrainingContext;

@SuppressWarnings({ "rawtypes", "unchecked" })
public class DeltaFlushBuffer implements VectorLogic<DeltaFlushBuffer> {
  
  private static final Logger log = LoggerFactory.getLogger(TrainingContext.class);
  
  public static DeltaValueAccumulator newAccumulator() {
    return new DeltaValueAccumulator1();
  }
  
  private final DeltaValueAccumulator[] buffer;
  private final double[] inner;
  private final NNLayer layer;
  private boolean normalize = false;
  private LogNumber[] calcVector;
  
  public DeltaFlushBuffer(final double[] values, final DeltaValueAccumulator[] array, final NNLayer layer) {
    this.inner = values;
    this.layer = layer;
    this.buffer = array;
  }
  
  public DeltaFlushBuffer(final double[] values, final NNLayer layer) {
    assert null != values;
    this.inner = values;
    this.layer = layer;
    this.buffer = new DeltaValueAccumulator1[values.length];
    Arrays.setAll(this.buffer, i -> DeltaFlushBuffer.newAccumulator());
  }
  
  public DeltaFlushBuffer(final NDArray values, final NNLayer layer) {
    this(values.getData(), layer);
  }
  
  @Override
  public DeltaFlushBuffer add(final DeltaFlushBuffer right) {
    return join(right, (l, r) -> l.add(r));
  }
  
  @Override
  public double dotProduct(final DeltaFlushBuffer right) {
    return sum(right, (l, r) -> l.logValue().multiply(r.logValue()).doubleValue());
  }
  
  public void feed(final double[] data) {
    feed(new NDArray(new int[] { data.length }, data).log().getData());
  }
  
  public void feed(final LogNumber[] data) {
    assert(null == calcVector);
    final int dim = length();
    for (int i = 0; i < dim; i++) {
      final DeltaValueAccumulator prev = this.buffer[i];
      this.buffer[i] = prev.add(data[i]);
    }
  }
  
  public String getId() {
    if (null == this.layer) return "";
    return this.layer.getId();
  }
  
  public DeltaFlushBuffer getVector(final double fraction) {
    return this;
  }
  
  public boolean isFrozen() {
    return false;
  }
  
  protected <T extends DeltaValueAccumulator<T>> DeltaFlushBuffer join(final DeltaFlushBuffer right, final BiFunction<T, T, T> joiner) {
    return new DeltaFlushBuffer(this.inner, IntStream.range(0, this.buffer.length).mapToObj(i -> {
      final T l = (T) this.buffer[i];
      final T r = (T) right.buffer[i];
      if (null != l && null != r) return joiner.apply(l, r);
      if (null != l) return r;
      if (null != r) return l;
      return null;
    }).toArray(i -> new DeltaValueAccumulator[i]), this.layer);
  }
  
  @Override
  public double l1() {
    return Math.sqrt(Arrays.stream(this.buffer).mapToDouble(v -> {
      final double l2 = v.doubleValue();
      return l2 * l2;
    }).sum());
  }
  
  @Override
  public double l2() {
    return Math.sqrt(Arrays.stream(this.buffer).mapToDouble(v -> {
      final double l2 = v.doubleValue();
      return l2 * l2;
    }).sum());
  }
  
  public int length() {
    return this.inner.length;
  }
  
  protected <T extends DeltaValueAccumulator<T>> DeltaFlushBuffer map(final Function<T, T> mapper) {
    return new DeltaFlushBuffer(this.inner, Arrays.stream(this.buffer).map(x -> mapper.apply((T) x)).toArray(i -> new DeltaValueAccumulator[i]), this.layer);
  }
  
  @Override
  public DeltaFlushBuffer scale(final double f) {
    return map(x -> x.multiply(f));
  }
  
  protected <T extends DeltaValueAccumulator<T>> double sum(final DeltaFlushBuffer right, final BiFunction<T, T, Double> joiner) {
    return IntStream.range(0, this.buffer.length).mapToDouble(i -> {
      final T l = (T) this.buffer[i];
      final T r = (T) right.buffer[i];
      if (null != l && null != r) return joiner.apply(l, r);
      return 0;
    }).sum();
  }
  
  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append("DeltaFlushBuffer [");
    builder.append(Arrays.toString(this.buffer));
    builder.append("]");
    return builder.toString();
  }
  
  public synchronized void write(final double factor) {
    if (null == this.calcVector) {
      this.calcVector = calcVector();
    }
    final LogNumber[] cpy = Arrays.copyOf(calcVector(), calcVector.length);
    for (int i = 0; i < this.buffer.length; i++) {
      cpy[i] = cpy[i].multiply(factor);
    }
    if (this.layer.isVerbose()) {
      log.debug(String.format("Write to memory: %s", Arrays.toString(cpy)));
    }
    final int dim = length();
    if (null == cpy) return;
    for (int i = 0; i < dim; i++) {
      if (null == cpy[i]) {
        continue;
      }
      this.inner[i] += cpy[i].doubleValue();
      if (!Double.isFinite(this.inner[i])) {
        this.inner[i] = 0;
      }
    }
  }

  public LogNumber[] calcVector() {
    final LogNumber[] v = new LogNumber[this.buffer.length];
    for (int i = 0; i < this.buffer.length; i++) {
      v[i] = this.buffer[i].logValue();
    }
    if (isNormalize()) {
      LogNumber normalizationFactor = Stream.of(this.buffer).map(x -> x.logValue().abs()).max(Comparator.naturalOrder()).get();
      for (int i = 0; i < this.buffer.length; i++) {
        v[i] = v[i].divide(normalizationFactor);
      }
    }
    return v;
  }
  
  public boolean isNormalize() {
    return normalize;
  }
  
  public DeltaFlushBuffer setNormalize(boolean normalize) {
    this.normalize = normalize;
    return this;
  }
  
}
