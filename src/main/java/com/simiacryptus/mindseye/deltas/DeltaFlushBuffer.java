package com.simiacryptus.mindseye.deltas;

import java.util.Arrays;
import java.util.UUID;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.math.LogNumber;
import com.simiacryptus.mindseye.math.LogNumberVector;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.math.NumberVector;
import com.simiacryptus.mindseye.math.VectorLogic;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.training.TrainingContext;

@SuppressWarnings({ "rawtypes", "unchecked" })
public class DeltaFlushBuffer implements VectorLogic<DeltaFlushBuffer> {

  private static final Logger log = LoggerFactory.getLogger(TrainingContext.class);

  public static DeltaValueAccumulator newAccumulator() {
    return new DeltaValueAccumulator1();
  }

  private final DeltaValueAccumulator[] buffer;
  private double[] calcVector;
  private final NNLayer layer;
  double entropyDecayRate = 0.;
  private boolean normalize = false;

  private final double[] target;

  public DeltaFlushBuffer(final double[] values, final DeltaValueAccumulator[] array, final NNLayer layer) {
    this.target = values;
    this.layer = layer;
    this.buffer = array;
  }

  public DeltaFlushBuffer(final double[] values, final NNLayer layer) {
    assert null != values;
    this.target = values;
    this.layer = layer;
    this.buffer = new DeltaValueAccumulator1[values.length];
    Arrays.setAll(this.buffer, i -> DeltaFlushBuffer.newAccumulator());
  }

  @Override
  public DeltaFlushBuffer add(final DeltaFlushBuffer right) {
    return join(right, (l, r) -> l.add(r));
  }

  public double[] calcVector() {
    
    NumberVector state = new NumberVector(this.layer.state().stream().flatMapToDouble(x->Arrays.stream(x)).toArray());
    LogNumberVector v = new LogNumberVector(Arrays.stream(this.buffer).map(x -> x.logValue()).toArray(i -> new LogNumber[i]));
    if (isNormalize()) {
      //v = v.scale(1. / v.l2());
      v = v.scale(1. / state.l1());
    }
    NumberVector returnValue = v.toDouble();

    if (0 < this.entropyDecayRate) {
      final NumberVector   unitv = returnValue.unitV();
      NumberVector l1Decay = new NumberVector(this.target).scale(-this.entropyDecayRate);
      final double dotProduct = l1Decay.dotProduct(unitv);
      if (dotProduct < 0) {
        l1Decay = l1Decay.add(unitv.scale(-dotProduct));
      }
      //l1Decay = l1Decay.scale(v.l2());
      //returnValue = returnValue.add(l1Decay); 
    }
    return returnValue.getArray();
  }

  @Override
  public double dotProduct(final DeltaFlushBuffer right) {
    return sum(right, (l, r) -> l.logValue().multiply(r.logValue()).doubleValue());
  }

  public void feed(final double[] data) {
    feed(new NDArray(new int[] { data.length }, data).log().getData());
  }

  public void feed(final LogNumber[] data) {
    assert null == this.calcVector;
    final int dim = length();
    for (int i = 0; i < dim; i++) {
      final DeltaValueAccumulator prev = this.buffer[i];
      this.buffer[i] = prev.add(data[i]);
    }
  }

  public UUID getId() {
    return this.layer.getId();
  }

  public DeltaFlushBuffer getVector(final double fraction) {
    return this;
  }

  public boolean isFrozen() {
    return false;
  }

  public boolean isNormalize() {
    return this.normalize;
  }

  protected <T extends DeltaValueAccumulator<T>> DeltaFlushBuffer join(final DeltaFlushBuffer right, final BiFunction<T, T, T> joiner) {
    return new DeltaFlushBuffer(this.target, IntStream.range(0, this.buffer.length).mapToObj(i -> {
      final T l = (T) this.buffer[i];
      final T r = (T) right.buffer[i];
      if (null != l && null != r)
        return joiner.apply(l, r);
      if (null != l)
        return r;
      if (null != r)
        return l;
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
    return this.target.length;
  }

  protected <T extends DeltaValueAccumulator<T>> DeltaFlushBuffer map(final Function<T, T> mapper) {
    return new DeltaFlushBuffer(this.target, Arrays.stream(this.buffer).map(x -> mapper.apply((T) x)).toArray(i -> new DeltaValueAccumulator[i]), this.layer);
  }

  @Override
  public DeltaFlushBuffer scale(final double f) {
    return map(x -> x.multiply(f));
  }

  public DeltaFlushBuffer setNormalize(final boolean normalize) {
    this.normalize = normalize;
    return this;
  }

  protected <T extends DeltaValueAccumulator<T>> double sum(final DeltaFlushBuffer right, final BiFunction<T, T, Double> joiner) {
    return IntStream.range(0, this.buffer.length).mapToDouble(i -> {
      final T l = (T) this.buffer[i];
      final T r = (T) right.buffer[i];
      if (null != l && null != r)
        return joiner.apply(l, r);
      return 0;
    }).sum();
  }

  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append(getClass().getSimpleName());
    builder.append("/");
    builder.append(layer.getClass().getSimpleName());
    builder.append("/");
    builder.append(layer.getId());
    builder.append(" ");
    builder.append(Arrays.toString(this.getCalcVector()));
    builder.append(" -> ");
    builder.append(this.layer.state().stream().map(x->Arrays.toString(x)).reduce((a,b)->a+","+b).get());
    return builder.toString();
  }

  public synchronized void write(final double factor) {
    double[] calcVector = this.getCalcVector();
    calcVector = Arrays.copyOf(calcVector, calcVector.length);
    for (int i = 0; i < this.buffer.length; i++) {
      calcVector[i] = calcVector[i] * factor;
    }
    if (this.layer.isVerbose()) {
      log.debug(String.format("Write to memory: %s", Arrays.toString(calcVector)));
    }
    write(calcVector);
  }

  private void write(final double[] cpy) {
    final int dim = length();
    if (null == cpy)
      return;
    for (int i = 0; i < dim; i++) {
      this.target[i] += cpy[i];
      if (!Double.isFinite(this.target[i])) {
        this.target[i] = 0;
      }
    }
  }

  public double[] getCalcVector() {
    if (null == this.calcVector) {
      this.calcVector = calcVector();
    }
    return calcVector;
  }

}
