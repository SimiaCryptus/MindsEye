package com.simiacryptus.mindseye.data;

import java.util.function.Function;

import com.simiacryptus.util.Util;

public final class Simple2DCircle implements Function<Void, double[]> {

  private double[] center;
  private double radius;

  public Simple2DCircle(final double radius, final double[] center) {
    super();
    assert center.length == 2;
    this.radius = radius;
    this.center = center;
  }

  @Override
  public double[] apply(final Void n) {
    final double x = Util.R.get().nextDouble() * 2 * Math.PI;
    return new double[] { Math.sin(x) * this.radius + this.center[0], Math.cos(x) * this.radius + this.center[1] };
  }
}
