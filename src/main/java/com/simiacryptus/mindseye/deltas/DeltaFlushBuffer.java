package com.simiacryptus.mindseye.deltas;

import java.util.Arrays;
import java.util.Comparator;
import java.util.TreeSet;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.math.LogNumber;
import com.simiacryptus.mindseye.math.NDArray;

public class DeltaFlushBuffer implements DeltaSink, VectorLogic<DeltaFlushBuffer> {

  public static class DeltaValueAccumulator {
    public TreeSet<LogNumber> numbers = new TreeSet<>();

    public DeltaValueAccumulator() {
    }
    public DeltaValueAccumulator(DeltaValueAccumulator toCopy) {
      numbers.addAll(toCopy.numbers);
    }
    public synchronized LogNumber logValue() {
      LogNumber[] array = numbers.stream().toArray(i->new LogNumber[i]);
      return array[array.length/2];
    }
    public synchronized DeltaValueAccumulator add(LogNumber r) {
      numbers.add(r);
      return this;
    }
    public DeltaValueAccumulator add(DeltaValueAccumulator r) {
      DeltaValueAccumulator copy = new DeltaValueAccumulator(this);
      copy.numbers.addAll(r.numbers);
      return copy;
    }

    public DeltaValueAccumulator multiply(double r) {
      return map(x->x.multiply(r));
    }

    private synchronized DeltaValueAccumulator map(Function<LogNumber,LogNumber> f) {
      DeltaValueAccumulator copy = new DeltaValueAccumulator();
      numbers.stream().map(f).forEach(x->copy.add(x));
      return copy;
    }
    public double doubleValue() {
      return logValue().doubleValue();
    }
    
  }
  
  private final DeltaValueAccumulator[] buffer;
  private final DeltaSink inner;
  private NNLayer layer;

  private LogNumber normalizationFactor;
  private LogNumber rate = LogNumber.log(1);
  private boolean reset = false;

  public DeltaFlushBuffer(final DeltaSink values) {
    this.inner = values;
    this.buffer = new DeltaValueAccumulator[values.length()];
    Arrays.setAll(this.buffer, i->new DeltaValueAccumulator());
  }

  public DeltaFlushBuffer(final double[] bias) {
    this(new DeltaMemoryWriter(bias));
  }

  protected DeltaFlushBuffer(final double[] ptr, final NNLayer layer) {
    super();
    this.inner = new DeltaMemoryWriter(ptr);
    this.buffer = new DeltaValueAccumulator[ptr.length];
    Arrays.setAll(this.buffer, i->new DeltaValueAccumulator());
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
      this.normalizationFactor = Stream.of(this.buffer).map(x->x.logValue().abs()).max(Comparator.naturalOrder()).get();
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
    return map(x->x.multiply(f));
  }

  @Override
  public double dotProduct(DeltaFlushBuffer right) {
    return sum(right, (l,r)-> l.logValue().multiply(r.logValue()).doubleValue());
  }

  @Override
  public DeltaFlushBuffer add(DeltaFlushBuffer right) {
    return join(right, (l,r)-> l.add(r));
  }

  protected DeltaFlushBuffer join(DeltaFlushBuffer right, BiFunction<DeltaValueAccumulator, DeltaValueAccumulator, DeltaValueAccumulator> joiner) {
    return new DeltaFlushBuffer(inner, IntStream.range(0, buffer.length).mapToObj(i->{
      DeltaValueAccumulator l = buffer[i];
      DeltaValueAccumulator r = right.buffer[i];
      if(null!=l&&null!=r) return joiner.apply(l, r);
      if(null!=l) return r;
      if(null!=r) return l;
      return null;
    }).toArray(i->new DeltaValueAccumulator[i]));
  }

  protected double sum(DeltaFlushBuffer right, BiFunction<DeltaValueAccumulator, DeltaValueAccumulator, Double> joiner) {
    return IntStream.range(0, buffer.length).mapToDouble(i->{
      DeltaValueAccumulator l = buffer[i];
      DeltaValueAccumulator r = right.buffer[i];
      if(null!=l&&null!=r) return joiner.apply(l, r);
      return 0;
    }).sum();
  }

  protected DeltaFlushBuffer map(Function<DeltaValueAccumulator, DeltaValueAccumulator> mapper) {
    return new DeltaFlushBuffer(inner, Arrays.stream(buffer).map(mapper).toArray(i->new DeltaValueAccumulator[i]));
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
