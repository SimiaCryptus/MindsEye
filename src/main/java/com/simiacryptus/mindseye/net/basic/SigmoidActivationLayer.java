package com.simiacryptus.mindseye.net.basic;

public final class SigmoidActivationLayer extends SimpleActivationLayer<SigmoidActivationLayer> {

  /**
   * 
   */
  private static final long serialVersionUID = -1676818127036480927L;
  private static final double MIN_X = -20;
  private static final double MAX_X = -MIN_X;
  private static final double MIN_F = Math.exp(MIN_X);
  private static final double MAX_F = Math.exp(MAX_X);

  private boolean balanced = true;

  public SigmoidActivationLayer() {
  }

  @Override
  protected final void eval(final double x, final double[] results) {
    final double minDeriv = 0;
    final double ex = exp(x);
    final double ex1 = 1 + ex;
    double d = ex / (ex1 * ex1);
    double f = 1 / (1 + 1. / ex);
    // double d = f * (1 - f);
    if (!Double.isFinite(d) || d < minDeriv) {
      d = minDeriv;
    }
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    if (isBalanced()) {
      d = 2 * d;
      f = 2 * f - 1;
    }
    results[0] = f;
    results[1] = d;
  }

  private double exp(final double x) {
    if (x < MIN_X)
      return MIN_F;
    if (x > MAX_X)
      return MAX_F;
    return Math.exp(x);
  }

  public boolean isBalanced() {
    return this.balanced;
  }

  public SigmoidActivationLayer setBalanced(final boolean balanced) {
    this.balanced = balanced;
    return this;
  }
}
