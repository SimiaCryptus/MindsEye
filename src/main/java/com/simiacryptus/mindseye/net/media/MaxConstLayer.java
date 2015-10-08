package com.simiacryptus.mindseye.net.media;

import com.simiacryptus.mindseye.net.basic.SimpleActivationLayer;

@SuppressWarnings("serial")
public class MaxConstLayer extends SimpleActivationLayer<MaxConstLayer> {

  private double value = 0;
  
  @Override
  protected void eval(double x, double[] results) {
    double d = x<value?0:1;
    double f = x<value?value:x;
    assert Double.isFinite(d);
    results[0] = f;
    results[1] = d;
  }

  public double getValue() {
    return value;
  }

  public MaxConstLayer setValue(double value) {
    this.value = value;
    return this;
  }
}
