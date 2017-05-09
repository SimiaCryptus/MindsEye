package com.simiacryptus.mindseye.net;

import java.util.Arrays;
import java.util.UUID;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@SuppressWarnings({ "rawtypes", "unchecked" })
public class DeltaBuffer {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DeltaBuffer.class);

  public final double[] delta;
  public final NNLayer layer;
  public final double[] target;

  public DeltaBuffer(final double[] values, final double[] array, final NNLayer layer) {
    this.target = values;
    this.layer = layer;
    this.delta = array;
  }

  public DeltaBuffer(final double[] values, final NNLayer layer) {
    assert null != values;
    this.target = values;
    this.layer = layer;
    this.delta = new double[values.length];
    Arrays.setAll(this.delta, i -> 0);
  }

  public DeltaBuffer accumulate(final double[] data) {
    final int dim = length();
    for (int i = 0; i < dim; i++) {
      final double prev = this.delta[i];
      this.delta[i] = prev + data[i];
    }
    return this;
  }

  public double[] copyDelta() {
    return null==delta?null:Arrays.copyOf(delta,delta.length);
  }

  public double[] copyTarget() {
    return null==target?null:Arrays.copyOf(target,target.length);
  }

  public UUID getId() {
    return this.layer.getId();
  }

  public DeltaBuffer getVector(final double fraction) {
    return this;
  }

  public boolean isFrozen() {
    return false;
  }

  public int length() {
    return this.target.length;
  }

  protected DeltaBuffer map(final java.util.function.DoubleUnaryOperator mapper) {
    return new DeltaBuffer(this.target, Arrays.stream(this.delta).map(x -> mapper.applyAsDouble(x)).toArray(), this.layer);
  }

  public DeltaBuffer scale(final double f) {
    return map(x -> x * f);
  }

  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append(getClass().getSimpleName());
    builder.append("/");
    builder.append(this.layer.getClass().getSimpleName());
    builder.append("/");
    builder.append(this.layer.getId());
    builder.append(" ");
    builder.append(Arrays.toString(this.delta));
    builder.append(" -> ");
    builder.append(this.layer.state().stream().map(x -> Arrays.toString((double[]) x)).reduce((a, b) -> a + "," + b).get());
    return builder.toString();
  }

  public synchronized final void write(final double factor) {
    double[] calcVector = this.delta;
    if (null == calcVector)
      return;
    calcVector = Arrays.copyOf(calcVector, calcVector.length);
    for (int i = 0; i < this.delta.length; i++) {
      calcVector[i] = calcVector[i] * factor;
    }
    final int dim = length();
    for (int i = 0; i < dim; i++) {
      this.target[i] = this.target[i] + calcVector[i];
    }
  }

  public double dot(DeltaBuffer right) {
    assert(this.target == right.target);
    assert(this.delta.length == right.delta.length);
    return IntStream.range(0, this.delta.length).mapToDouble(i-> delta[i]*right.delta[i]).sum();
  }

  public double sum() {
    return Arrays.stream(this.delta).sum();
  }

  public double sumSq() {
    return Arrays.stream(this.delta).map(x->x*x).sum();
  }

}
