package com.simiacryptus.mindseye.net.basic;

public final class AbsActivationLayer extends SimpleActivationLayer<AbsActivationLayer> {

  /**
   * 
   */
  private static final long serialVersionUID = -5520500379591109767L;

  public AbsActivationLayer() {
  }

  @Override
  protected final void eval(final double x, final double[] results) {
    final double minDeriv = 0;
    final double d = x < 0 ? -1 : 1;
    final double f = x < 0 ? -x : x;
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    results[0] = f;
    results[1] = d;
  }

}
