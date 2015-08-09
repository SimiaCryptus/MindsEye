package com.simiacryptus.mindseye.learning;

import com.simiacryptus.mindseye.NDArray;

public class DeltaNormalizer implements DeltaSink {

  private double alpha = 2.;
  private double beta = 4.;
  private boolean enabled = false;
  private DeltaSink sink;

  private double[] values;

  public DeltaNormalizer() {
    super();
  }

  public DeltaNormalizer(final DeltaSink sink, final double[] values) {
    this.sink = sink;
    this.values = values;
  }

  public DeltaNormalizer(final double[] values) {
    this(new DeltaMemoryWriter(values), values);
  }

  public DeltaNormalizer(final NDArray values) {
    this(values.getData());
  }

  @Override
  public void feed(final double[] data) {

    final int dim = length();
    final double[] v = new double[data.length];
    for (int i = 0; i < dim; i++) {
      double f;
      if (this.enabled) {
        final double before = Math.log(Math.abs(this.values[i]) / this.alpha) * this.beta;
        final double after = Math.log(Math.abs(this.values[i] + data[i]) / this.alpha) * this.beta;
        f = before > after ? 1 : Math.min(Math.abs(1. / (Math.exp(after) - Math.exp(before))), 1.);
        assert Double.isFinite(f);
        assert f <= 1.0;
      }
      else
      {
        f = 1.;
      }
      v[i] += f * data[i];
    }
    this.sink.feed(v);
  }

  public double getAlpha() {
    return this.alpha;
  }

  public double getBeta() {
    return this.beta;
  }

  public boolean isEnabled() {
    return this.enabled;
  }

  @Override
  public int length() {
    return this.values.length;
  }

  public DeltaNormalizer setAlpha(final double alpha) {
    this.alpha = alpha;
    return this;
  }

  public DeltaNormalizer setBeta(final double beta) {
    this.beta = beta;
    return this;
  }

  public DeltaNormalizer setEnabled(final boolean enabled) {
    this.enabled = enabled;
    return this;
  }

}
