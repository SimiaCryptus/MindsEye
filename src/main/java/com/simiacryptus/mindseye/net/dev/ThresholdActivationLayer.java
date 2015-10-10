package com.simiacryptus.mindseye.net.dev;

import com.simiacryptus.mindseye.net.basic.SimpleActivationLayer;

public final class ThresholdActivationLayer extends SimpleActivationLayer<ThresholdActivationLayer> {

  private static final long serialVersionUID = 7627314158757648516L;

  private double factor = -1;

  private int polarity = -1;
  private double threshold = 0;

  public ThresholdActivationLayer() {
  }

  @Override
  protected final void eval(final double x, final double[] results) {
    final boolean inrange = Double.compare(x, getThreshold()) == getPolarity();
    final double d = inrange ? getFactor() : 0;
    final double f = (inrange ? x : getThreshold()) * getFactor();
    assert Double.isFinite(d);
    if (0 > f) {
      assert 0 <= f;
    }
    results[0] = f;
    results[1] = d;
  }

  public double getFactor() {
    return this.factor;
  }

  public int getPolarity() {
    return this.polarity;
  }

  public double getThreshold() {
    return this.threshold;
  }

  public ThresholdActivationLayer setFactor(final double factor) {
    this.factor = factor;
    return this;
  }

  public ThresholdActivationLayer setPolarity(final int polarity) {
    this.polarity = polarity;
    return this;
  }

  public ThresholdActivationLayer setThreshold(final double threshold) {
    this.threshold = threshold;
    return this;
  }

}
