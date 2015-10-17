package com.simiacryptus.mindseye.core.delta;

import java.util.Arrays;
import java.util.UUID;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.core.TrainingContext;

@SuppressWarnings({ "rawtypes", "unchecked" })
public class DeltaBuffer {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(TrainingContext.class);

  public static double newAccumulator() {
    return 0;
  }

  private final double[] buffer;
  private double[] calcVector;
  private final NNLayer layer;

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

  private double[] calcVector() {
    return new NumberVector(this.buffer).getArray();
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

  public int length() {
    return this.target.length;
  }

  protected DeltaBuffer map(final java.util.function.DoubleUnaryOperator mapper) {
    return new DeltaBuffer(this.target, Arrays.stream(this.buffer).map(x -> mapper.applyAsDouble(x)).toArray(), this.layer);
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
