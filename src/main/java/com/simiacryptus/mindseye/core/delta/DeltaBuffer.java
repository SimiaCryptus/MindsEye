package com.simiacryptus.mindseye.core.delta;

import java.util.Arrays;
import java.util.UUID;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.core.TrainingContext;

@SuppressWarnings({ "rawtypes", "unchecked" })
public class DeltaBuffer implements VectorLogic<DeltaBuffer> {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(TrainingContext.class);

  public static double newAccumulator() {
    return 0;
  }

  private final double[] buffer;
  private double[] calcVector;
  double entropyDecayRate = 0.0;
  private final NNLayer layer;
  private boolean normalize = false;

  public final double[] target;

  public DeltaBuffer(final double[] values, final double[] array, final NNLayer layer) {
    this.target = values;
    this.layer = layer;
    this.buffer = array;
  }

  public DeltaBuffer(final double[] values, final NNLayer layer) {
    assert null != values;
    this.target = values;
    this.layer = layer;
    this.buffer = new double[values.length];
    Arrays.setAll(this.buffer, i -> DeltaBuffer.newAccumulator());
  }

  @Override
  public DeltaBuffer add(final DeltaBuffer right) {
    return join(right, (l, r) -> l + r);
  }

  private double[] calcVector() {

    final NumberVector state = new NumberVector(this.layer.state().stream().flatMapToDouble(x -> Arrays.stream((double[]) x)).toArray());
    NumberVector returnValue = new NumberVector(this.buffer);
    if (isNormalize()) {
      // v = v.scale(1. / v.l2());
      returnValue = returnValue.scale(1. / state.l1());
    }
    if (0 < this.entropyDecayRate) {
      final NumberVector unitv = returnValue.unitV();
      NumberVector l1Decay = new NumberVector(this.target).scale(-this.entropyDecayRate);
      final double dotProduct = l1Decay.dotProduct(unitv);
      if (dotProduct < 0) {
        l1Decay = l1Decay.add(unitv.scale(-dotProduct));
      }
      // l1Decay = l1Decay.scale(v.l2());
      // returnValue = returnValue.add(l1Decay);
    }
    return returnValue.getArray();
  }

  @Override
  public double dotProduct(final DeltaBuffer right) {
    return sum(right, (l, r) -> l * r);
  }

  private Double f(final DeltaBuffer right, final java.util.function.DoubleBinaryOperator joiner, final int i) {
    final double l = this.buffer[i];
    final double r = right.buffer[i];
    return joiner.applyAsDouble(l, r);
  }

  public void feed(final double[] data) {
    assert null == this.calcVector;
    final int dim = length();
    for (int i = 0; i < dim; i++) {
      final double prev = this.buffer[i];
      this.buffer[i] = prev + data[i];
    }
  }

  public double[] getCalcVector() {
    if (null == this.calcVector) {
      this.calcVector = calcVector();
    }
    return this.calcVector;
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

  public boolean isNormalize() {
    return this.normalize;
  }

  protected DeltaBuffer join(final DeltaBuffer right, final java.util.function.DoubleBinaryOperator joiner) {
    return new DeltaBuffer(this.target, IntStream.range(0, this.buffer.length).mapToDouble(i -> {
      return f(right, joiner, i);
    }).toArray(), this.layer);
  }

  @Override
  public double l1() {
    return Math.sqrt(Arrays.stream(this.buffer).map(v -> v * v).sum());
  }

  @Override
  public double l2() {
    return Math.sqrt(Arrays.stream(this.buffer).map(v -> v * v).sum());
  }

  public int length() {
    return this.target.length;
  }

  protected DeltaBuffer map(final java.util.function.DoubleUnaryOperator mapper) {
    return new DeltaBuffer(this.target, Arrays.stream(this.buffer).map(x -> mapper.applyAsDouble(x)).toArray(), this.layer);
  }

  @Override
  public DeltaBuffer scale(final double f) {
    return map(x -> x * f);
  }

  public DeltaBuffer setNormalize(final boolean normalize) {
    this.normalize = normalize;
    return this;
  }

  protected double sum(final DeltaBuffer right, final java.util.function.DoubleBinaryOperator joiner) {
    return IntStream.range(0, this.buffer.length).mapToDouble(i -> {
      final double l = this.buffer[i];
      final double r = right.buffer[i];
      return joiner.applyAsDouble(l, r);
    }).sum();
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
    builder.append(Arrays.toString(getCalcVector()));
    builder.append(" -> ");
    builder.append(this.layer.state().stream().map(x -> Arrays.toString((double[]) x)).reduce((a, b) -> a + "," + b).get());
    return builder.toString();
  }

  public synchronized final void write(final double factor) {
    double[] calcVector = getCalcVector();
    if (null == calcVector)
      return;
    calcVector = Arrays.copyOf(calcVector, calcVector.length);
    for (int i = 0; i < this.buffer.length; i++) {
      calcVector[i] = calcVector[i] * factor;
    }
    final int dim = length();
    for (int i = 0; i < dim; i++) {
      this.target[i] = this.target[i] + calcVector[i];
    }
  }

}
