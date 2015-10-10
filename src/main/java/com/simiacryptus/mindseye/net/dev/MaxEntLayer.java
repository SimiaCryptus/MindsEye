package com.simiacryptus.mindseye.net.dev;

import com.simiacryptus.mindseye.net.basic.SimpleActivationLayer;

@SuppressWarnings("serial")
public class MaxEntLayer extends SimpleActivationLayer<MaxEntLayer> {

  double min = 1e-2;
  
  @Override
  protected void eval(double x, double[] results) {
    final double minDeriv = 0;
    double log = Math.log(Math.max(x, min));
    double d = -(1+log);
    double f = -x*log;
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    results[0] = f;
    results[1] = d;
  }
}
