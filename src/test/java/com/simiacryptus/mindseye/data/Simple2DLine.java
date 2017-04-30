package com.simiacryptus.mindseye.data;

import java.util.Random;
import java.util.function.Function;

import com.simiacryptus.util.Util;

public final class Simple2DLine implements Function<Void, double[]> {

  public final double bottom;
  public final double height;
  public final double left;
  public final double width;

  public Simple2DLine() {
    this(Util.R.get());
  }

  public Simple2DLine(final double left, final double bottom, final double width, final double height) {
    super();
    this.width = width;
    this.left = left;
    this.height = height;
    this.bottom = bottom;
  }

  public Simple2DLine(final double[]... pts) {
    super();
    assert pts.length == 2;
    this.width = pts[0][0] - pts[1][0];
    this.left = pts[1][0];
    this.height = pts[0][1] - pts[1][1];
    this.bottom = pts[1][1];
  }

  public Simple2DLine(final Random random) {
    super();
    final Random r = random;
    this.width = r.nextGaussian() * 4;
    this.left = r.nextGaussian() - 2;
    this.height = r.nextGaussian() * 4;
    this.bottom = r.nextGaussian() - 2;
  }

  @Override
  public double[] apply(final Void n) {
    final double x = Util.R.get().nextDouble();
    return new double[] { this.width * x + this.left, this.height * x + this.bottom };
  }
}
