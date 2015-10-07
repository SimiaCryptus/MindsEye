package com.simiacryptus.mindseye.net.basic;

public final class SqActivationLayer extends SimpleActivationLayer<SqActivationLayer> {


  /**
   * 
   */
  private static final long serialVersionUID = -5520500379591109767L;

  public SqActivationLayer() {
  }

  @Override
  protected final void eval(final double x, final double[] results) {
    final double minDeriv = 0;
    double d = 2*x;
    double f = x*x;
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    results[0] = f;
    results[1] = d;
  }

}
