package com.simiacryptus.mindseye.net.media;

import com.simiacryptus.mindseye.net.activation.SimpleActivationLayer;

@SuppressWarnings("serial")
public class MaxEntLayer extends SimpleActivationLayer<MaxEntLayer> {

  @Override
  protected void eval(final double x, final double[] results) {
    final double minDeriv = 0;
    double d;
    double f;
    if (0. == x) {
      d = 0;
      f = 0;
    } else {
      final double log = Math.log(Math.abs(x));
      d = -(1 + log);
      f = -x * log;
    }
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    results[0] = f;
    results[1] = d;
  }
}
