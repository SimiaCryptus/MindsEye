package com.simiacryptus.mindseye.net.dev;

import com.simiacryptus.mindseye.net.basic.SimpleActivationLayer;

public final class ThresholdActivationLayer extends SimpleActivationLayer<ThresholdActivationLayer> {

  private static final long serialVersionUID = 7627314158757648516L;

  public ThresholdActivationLayer() {
  }

  private double threshold = 0;
  private double factor = -1;
  private int polarity = -1;
  
  @Override
  protected final void eval(final double x, final double[] results) {
    boolean inrange = Double.compare(x,getThreshold())==getPolarity();
    double d = inrange?getFactor():0;
    double f = (inrange?x:getThreshold())*getFactor();
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

  public ThresholdActivationLayer setThreshold(double threshold) {
    this.threshold = threshold;
    return this;
  }

  public double getFactor() {
    return factor;
  }

  public ThresholdActivationLayer setFactor(double factor) {
    this.factor = factor;
    return this;
  }

  public int getPolarity() {
    return polarity;
  }

  public ThresholdActivationLayer setPolarity(int polarity) {
    this.polarity = polarity;
    return this;
  }

}
