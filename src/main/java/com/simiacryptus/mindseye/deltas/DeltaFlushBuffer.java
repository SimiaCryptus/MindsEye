package com.simiacryptus.mindseye.deltas;

import java.util.Arrays;
import java.util.Comparator;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.math.LogNumber;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.TrainingContext;

@SuppressWarnings({ "rawtypes", "unchecked", "unused" })
public class DeltaFlushBuffer implements DeltaSink, VectorLogic<DeltaFlushBuffer> {
  
  private static final Logger log = LoggerFactory.getLogger(TrainingContext.class);
  
  public static DeltaValueAccumulator newAccumulator() {
    return new DeltaValueAccumulator1();
  }
  
  private final DeltaValueAccumulator[] buffer;
  private final DeltaSink inner;
  private final NNLayer layer;
  private LogNumber normalizationFactor;
  private LogNumber rate = LogNumber.log(1);
  
  private boolean reset = false;
  
  public DeltaFlushBuffer(final DeltaSink values, NNLayer layer) {
    assert null != values;
    this.inner = values;
    this.layer = layer;
    this.buffer = new DeltaValueAccumulator1[values.length()];
    Arrays.setAll(this.buffer, i -> DeltaFlushBuffer.newAccumulator());
  }
  
  public DeltaFlushBuffer(final DeltaSink values, final DeltaValueAccumulator[] array, NNLayer layer) {
    this.inner = values;
    this.layer = layer;
    this.buffer = array;
  }
  
  protected DeltaFlushBuffer(final double[] ptr, final NNLayer layer) {
    super();
    this.inner = new DeltaMemoryWriter(ptr);
    this.buffer = new DeltaValueAccumulator1[ptr.length];
    Arrays.setAll(this.buffer, i -> DeltaFlushBuffer.newAccumulator());
    this.layer = layer;
  }
  
  public DeltaFlushBuffer(final NDArray values, NNLayer layer) {
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
  
  @Override
  public void feed(final LogNumber[] data) {
    if (this.reset) {
      this.reset = false;
      Arrays.fill(this.buffer, LogNumber.ZERO);
    }
    final int dim = length();
    for (int i = 0; i < dim; i++) {
      DeltaValueAccumulator prev = this.buffer[i];
      this.buffer[i] = prev.add(data[i]);
    }
  }
  
  public String getId() {
    if (null == this.layer) return "";
    return this.layer.getId();
  }
  
  public double getRate() {
    return this.rate.doubleValue();
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
    }).toArray(i -> new DeltaValueAccumulator[i]), layer);
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
  
  @Override
  public int length() {
    return this.inner.length();
  }
  
  protected <T extends DeltaValueAccumulator<T>> DeltaFlushBuffer map(final Function<T, T> mapper) {
    return new DeltaFlushBuffer(this.inner, Arrays.stream(this.buffer).map(x -> mapper.apply((T) x)).toArray(i -> new DeltaValueAccumulator[i]), layer);
  }
  
  @Override
  public DeltaFlushBuffer scale(final double f) {
    return map(x -> x.multiply(f));
  }
  
  public void setRate(final double rate) {
    this.rate = LogNumber.log(rate);
  }
  
  protected <T extends DeltaValueAccumulator<T>> double sum(final DeltaFlushBuffer right, final BiFunction<T, T, Double> joiner) {
    return IntStream.range(0, this.buffer.length).mapToDouble(i -> {
      final T l = (T) this.buffer[i];
      final T r = (T) right.buffer[i];
      if (null != l && null != r) return joiner.apply(l, r);
      return 0;
    }).sum();
  }
  
  public synchronized void write(final double factor) {
    final LogNumber[] cpy = new LogNumber[this.buffer.length];
    if (!this.reset) {
      this.normalizationFactor = Stream.of(this.buffer).map(x -> x.logValue().abs()).max(Comparator.naturalOrder()).get();
    }
    for (int i = 0; i < this.buffer.length; i++) {
      cpy[i] = this.buffer[i].logValue().multiply(factor).multiply(getRate())
          .divide(this.normalizationFactor);
    }
    if (this.layer.isVerbose()) {
      log.debug(String.format("Write to memory: %s", Arrays.toString(cpy)));
    }
    this.inner.feed(cpy);
    this.reset = true;
  }
  
  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("DeltaFlushBuffer [");
    builder.append(Arrays.toString(buffer));
    builder.append("]");
    return builder.toString();
  }
  
}
