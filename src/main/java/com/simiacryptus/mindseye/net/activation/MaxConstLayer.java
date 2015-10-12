package com.simiacryptus.mindseye.net.activation;

@SuppressWarnings("serial")
public class MaxConstLayer extends SimpleActivationLayer<MaxConstLayer> {

  private double value = 0;

  @Override
  protected void eval(final double x, final double[] results) {
    final double d = x < this.value ? 0 : 1;
    final double f = x < this.value ? this.value : x;
    assert Double.isFinite(d);
    results[0] = f;
    results[1] = d;
  }

  public double getValue() {
    return this.value;
  }

  public MaxConstLayer setValue(final double value) {
    this.value = value;
    return this;
  }
}
