package com.simiacryptus.mindseye.deltas;

import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.Stream;

import com.google.common.hash.Hashing;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.math.LogNumber;
import com.simiacryptus.mindseye.math.NDArray;

public class DeltaFlushBuffer implements DeltaSink {

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

}
