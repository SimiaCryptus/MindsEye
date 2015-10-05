package com.simiacryptus.mindseye.net.dev;

import com.simiacryptus.mindseye.net.basic.SimpleActivationLayer;

public final class MinActivationLayer extends SimpleActivationLayer<MinActivationLayer> {

  private static final long serialVersionUID = 7627314158757648516L;

  public MinActivationLayer() {
  }

  private double threshold = 0;
  private double factor = -1;
  
  @Override
  protected final void eval(final double x, final double[] results) {
    double d = x<getThreshold()?getFactor():0;
    double f = Math.min(x,getThreshold())*getFactor();
    assert Double.isFinite(d);
    if(0>f){
      assert(0<=f);
    }
    results[0] = f;
    results[1] = d;
  }

  public double getThreshold() {
    return threshold;
  }

  public MinActivationLayer setThreshold(double threshold) {
    this.threshold = threshold;
    return this;
  }

  public double getFactor() {
    return factor;
  }

  public MinActivationLayer setFactor(double factor) {
    this.factor = factor;
    return this;
  }

}
