package com.simiacryptus.mindseye.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.function.Function;

import com.simiacryptus.mindseye.util.Util;

@SuppressWarnings("serial")
public final class UnionDistribution extends ArrayList<Function<Void, double[]>>implements Function<Void, double[]> {

  public UnionDistribution() {
    super();
  }

  public UnionDistribution(final Collection<? extends Function<Void, double[]>> c) {
    super(c);
  }

  @SafeVarargs
  public UnionDistribution(final Function<Void, double[]>... c) {
    this(Arrays.asList(c));
  }

  @Override
  public double[] apply(final Void t) {
    return get(Util.R.get().nextInt(size())).apply(t);
  }
}