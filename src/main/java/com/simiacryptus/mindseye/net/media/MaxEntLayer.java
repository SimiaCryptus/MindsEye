package com.simiacryptus.mindseye.net.media;

import com.simiacryptus.mindseye.net.basic.SimpleActivationLayer;

@SuppressWarnings("serial")
public class MaxEntLayer extends SimpleActivationLayer<MaxEntLayer> {

  double min = 1e-5;
  
  @Override
  protected void eval(double x, double[] results) {
    final double minDeriv = 0;
    double abs = Math.abs(x);
    double d;
    double f;
    if(abs < min)
    {
      double log = Math.log(min);
      d = 0;
      f = -x*log;
    } else {
      double log = Math.log(abs);
      d = -(1+log);
      f = -x*log;
    }
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    results[0] = f;
    results[1] = d;
  }
}
