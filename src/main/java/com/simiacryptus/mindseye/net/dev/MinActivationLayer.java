package com.simiacryptus.mindseye.net.dev;

import com.simiacryptus.mindseye.net.basic.SimpleActivationLayer;

public final class MinActivationLayer extends SimpleActivationLayer<MinActivationLayer> {


  public MinActivationLayer() {
  }

  private double threshold = 0;
  private double factor = -1;
  
  @Override
  protected final void eval(final double x, final double[] results) {
    final double minDeriv = 0;
    double d = x<getThreshold()?getFactor():0;
    double f = Math.min(x,getThreshold())*getFactor();
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    results[0] = f;
    results[1] = d;
  }

  double getThreshold() {
    return threshold;
  }

  MinActivationLayer setThreshold(double threshold) {
    this.threshold = threshold;
    return this;
  }

  double getFactor() {
    return factor;
  }

  MinActivationLayer setFactor(double factor) {
    this.factor = factor;
    return this;
  }

}
