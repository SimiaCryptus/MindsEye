package com.simiacryptus.mindseye.deltas;

import java.util.Arrays;
import java.util.Comparator;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import com.google.common.hash.Hashing;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.math.LogNumber;
import com.simiacryptus.mindseye.math.NDArray;

public class DeltaFlushBuffer implements DeltaSink, VectorLogic<DeltaFlushBuffer> {

  private final LogNumber[] buffer;
  private final DeltaSink inner;
  private NNLayer layer;

  private LogNumber normalizationFactor;
  private LogNumber rate = LogNumber.log(1);
  private boolean reset = false;

  public DeltaFlushBuffer(final DeltaSink values) {
    this.inner = values;
    this.buffer = new LogNumber[values.length()];
    Arrays.fill(this.buffer, LogNumber.ZERO);
  }

  public DeltaFlushBuffer(final double[] bias) {
    this(new DeltaMemoryWriter(bias));
  }

  protected DeltaFlushBuffer(final double[] ptr, final NNLayer layer) {
    super();
    this.inner = new DeltaMemoryWriter(ptr);
    this.buffer = new LogNumber[ptr.length];
    this.layer = layer;
  }

  public DeltaFlushBuffer(final NDArray values) {
    this(values.getData());
  }

  public DeltaFlushBuffer(final DeltaSink values, LogNumber[] array) {
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
      final LogNumber prev = this.buffer[i];
      this.buffer[i] = null == prev ? data[i] : prev.add(data[i]);
    }
  }

  public String getId() {
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

  public void write(final double factor) {
    write(factor, 1., 0l);
  }

  public synchronized void write(final double factor, final double fraction, final long mask) {
    final long longF = (long) (fraction * Long.MAX_VALUE);
    final LogNumber[] cpy = new LogNumber[this.buffer.length];

    if (!this.reset) {
      this.normalizationFactor = Stream.of(this.buffer).map(LogNumber::abs).max(Comparator.naturalOrder()).get();
    }

    for (int i = 0; i < this.buffer.length; i++) {
      if (fraction < 1.) {
        final long hash = Hashing.sha1().hashLong(i ^ mask).asLong();
        if (longF > hash) {
          continue;
        }
      }
      cpy[i] = this.buffer[i].multiply(factor).multiply(getRate())
          .divide(this.normalizationFactor);
    }
    this.inner.feed(cpy);
    this.reset = true;
  }

  @Override
  public DeltaFlushBuffer scale(double f) {
    return map(x->x.multiply(f));
  }

  @Override
  public double dotProduct(DeltaFlushBuffer right) {
    return sum(right, (l,r)-> l.multiply(r).doubleValue());
  }

  @Override
  public DeltaFlushBuffer add(DeltaFlushBuffer right) {
    return join(right, (l,r)-> l.add(r));
  }

  public DeltaFlushBuffer join(DeltaFlushBuffer right, BiFunction<LogNumber, LogNumber, LogNumber> joiner) {
    return new DeltaFlushBuffer(inner, IntStream.range(0, buffer.length).mapToObj(i->{
      LogNumber l = buffer[i];
      LogNumber r = right.buffer[i];
      if(null!=l&&null!=r) return joiner.apply(l, r);
      if(null!=l) return r;
      if(null!=r) return l;
      return null;
    }).toArray(i->new LogNumber[i]));
  }

  public double sum(DeltaFlushBuffer right, BiFunction<LogNumber, LogNumber, Double> joiner) {
    return IntStream.range(0, buffer.length).mapToDouble(i->{
      LogNumber l = buffer[i];
      LogNumber r = right.buffer[i];
      if(null!=l&&null!=r) return joiner.apply(l, r);
      return 0;
    }).sum();
  }

  public DeltaFlushBuffer map(Function<LogNumber, LogNumber> mapper) {
    return new DeltaFlushBuffer(inner, Arrays.stream(buffer).map(mapper).toArray(i->new LogNumber[i]));
  }

  @Override
  public double l1() {
    return Math.sqrt(Arrays.stream(buffer).mapToDouble(v->{
      double l2 = v.doubleValue();
      return l2*l2;
    }).sum());
  }
  @Override
  public double l2() {
    return Math.sqrt(Arrays.stream(buffer).mapToDouble(v->{
      double l2 = v.doubleValue();
      return l2*l2;
    }).sum());
  }
}
