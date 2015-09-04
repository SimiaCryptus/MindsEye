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

@SuppressWarnings({ "rawtypes", "unchecked", "unused" })
public class DeltaFlushBuffer implements DeltaSink, VectorLogic<DeltaFlushBuffer> {
  
  private static final Logger log = LoggerFactory.getLogger(TrainingContext.class);
  
  private final DeltaValueAccumulator[] buffer;
  private final DeltaSink inner;
  private NNLayer layer;
  private LogNumber normalizationFactor;
  private LogNumber rate = LogNumber.log(1);
  private boolean reset = false;
  
  public DeltaFlushBuffer(final DeltaSink values) {
    assert(null != values);
    this.inner = values;
    this.buffer = new DeltaValueAccumulator1[values.length()];
    Arrays.setAll(this.buffer, i -> newAccumulator());
  }
  
  public static DeltaValueAccumulator newAccumulator() {
    return new DeltaValueAccumulator1();
  }
  
  public DeltaFlushBuffer(final double[] bias) {
    this(new DeltaMemoryWriter(bias));
  }
  
  protected DeltaFlushBuffer(final double[] ptr, final NNLayer layer) {
    super();
    this.inner = new DeltaMemoryWriter(ptr);
    this.buffer = new DeltaValueAccumulator1[ptr.length];
    Arrays.setAll(this.buffer, i -> newAccumulator());
    this.layer = layer;
  }
  
  public DeltaFlushBuffer(final NDArray values) {
    this(values.getData());
  }
  
  public DeltaFlushBuffer(final DeltaSink values, DeltaValueAccumulator[] array) {
    this.inner = values;
    this.buffer = array;
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
      this.buffer[i] = this.buffer[i].add(data[i]);
    }
  }
  
  public String getId() {
    if(null == this.layer) return "";
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
  
  @Override
  public int length() {
    return this.inner.length();
  }
  
  public void setRate(final double rate) {
    this.rate = LogNumber.log(rate);
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
    this.inner.feed(cpy);
    this.reset = true;
  }
  
  @Override
  public DeltaFlushBuffer scale(double f) {
    return map(x -> x.multiply(f));
  }
  
  @Override
  public double dotProduct(DeltaFlushBuffer right) {
    return sum(right, (l, r) -> l.logValue().multiply(r.logValue()).doubleValue());
  }
  
  @Override
  public DeltaFlushBuffer add(DeltaFlushBuffer right) {
    return join(right, (l, r) -> l.add(r));
  }
  
  protected <T extends DeltaValueAccumulator<T>> DeltaFlushBuffer join(DeltaFlushBuffer right, BiFunction<T, T, T> joiner) {
    return new DeltaFlushBuffer(inner, IntStream.range(0, buffer.length).mapToObj(i -> {
      T l = (T) buffer[i];
      T r = (T) right.buffer[i];
      if (null != l && null != r) return joiner.apply(l, r);
      if (null != l) return r;
      if (null != r) return l;
      return null;
    }).toArray(i -> new DeltaValueAccumulator[i]));
  }
  
  protected <T extends DeltaValueAccumulator<T>> double sum(DeltaFlushBuffer right, BiFunction<T, T, Double> joiner) {
    return IntStream.range(0, buffer.length).mapToDouble(i -> {
      T l = (T) buffer[i];
      T r = (T) right.buffer[i];
      if (null != l && null != r) return joiner.apply(l, r);
      return 0;
    }).sum();
  }
  
  protected <T extends DeltaValueAccumulator<T>> DeltaFlushBuffer map(Function<T, T> mapper) {
    return new DeltaFlushBuffer(inner, Arrays.stream(buffer).map(x -> mapper.apply((T) x)).toArray(i -> new DeltaValueAccumulator[i]));
  }
  
  @Override
  public double l1() {
    return Math.sqrt(Arrays.stream(buffer).mapToDouble(v -> {
      double l2 = v.doubleValue();
      return l2 * l2;
    }).sum());
  }
  
  @Override
  public double l2() {
    return Math.sqrt(Arrays.stream(buffer).mapToDouble(v -> {
      double l2 = v.doubleValue();
      return l2 * l2;
    }).sum());
  }
}
