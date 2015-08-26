package com.simiacryptus.mindseye.learning;

import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.google.common.hash.Hashing;
import com.simiacryptus.mindseye.math.LogNumber;
import com.simiacryptus.mindseye.math.NDArray;

public class DeltaFlushBuffer implements DeltaSink {

  private final LogNumber[] buffer;
  private final DeltaSink inner;
  private LogNumber rate = LogNumber.log(1);

  private boolean reset = false;
  private LogNumber normalizationFactor;
  
  protected DeltaFlushBuffer() {
    super();
    this.inner = null;
    this.buffer = new LogNumber[]{};
  }

  public DeltaFlushBuffer(final DeltaSink values) {
    this.inner = values;
    this.buffer = new LogNumber[values.length()];
    Arrays.fill(this.buffer, LogNumber.ZERO);
  }

  public DeltaFlushBuffer(final NDArray values) {
    this(values.getData());
  }

  public DeltaFlushBuffer(double[] bias) {
    this(new DeltaMemoryWriter(bias));
  }

  public void feed(final double[] data) {
    feed(new NDArray(new int[]{data.length}, data).log().getData());
  }

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

  public double getRate() {
    return this.rate.doubleValue();
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

  public synchronized void write(final double factor, double fraction, long mask) {
    long longF = (long)(fraction * Long.MAX_VALUE);
    final LogNumber[] cpy = new LogNumber[this.buffer.length];

    if(!this.reset) {
      this.normalizationFactor = Stream.of(this.buffer).map(LogNumber::abs).max(Comparator.naturalOrder()).get();
      this.normalizationFactor = LogNumber.ONE;
    }

    for (int i = 0; i < this.buffer.length; i++) {
      if(fraction<1.){
        long hash = Hashing.sha1().hashLong(i ^ mask).asLong();
        if(longF > (hash)){
          continue;
        }
      }
      cpy[i] = this.buffer[i].multiply(factor).multiply(getRate())
          .divide(this.normalizationFactor);
    }
    this.inner.feed(cpy);
    this.reset = true;
  }

  public DeltaFlushBuffer getVector(double fraction) {
    return this;
  }

}
