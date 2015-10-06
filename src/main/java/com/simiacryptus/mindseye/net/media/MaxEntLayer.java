package com.simiacryptus.mindseye.net.media;

import com.simiacryptus.mindseye.net.basic.SimpleActivationLayer;

public class MaxEntLayer extends SimpleActivationLayer<MaxEntLayer> {

  @Override
  protected void eval(double x, double[] results) {
    final double minDeriv = 0;
    double log = Math.log(x);
    double d = -(1+log);
    double f = -x*log;
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    results[0] = f;
    results[1] = d;
  }
}
