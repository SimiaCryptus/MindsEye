package com.simiacryptus.mindseye.deltas;

import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.IntStream;

import com.simiacryptus.mindseye.math.NDArray;

public abstract class NNResult {

  public final NDArray data;

  public NNResult(final NDArray data) {
    super();
    this.data = data;
  }

  public final NDArray delta(final int k) {
    return delta(ideal(k));
  }

  public final NDArray delta(final NDArray target) {
    assert this.data.dim() == target.dim();
    final NDArray delta = new NDArray(this.data.getDims());
    Arrays.parallelSetAll(delta.getData(), i -> (target.getData()[i] - NNResult.this.data.getData()[i]));
    return delta;
  }

  public final double errMisclassification(final int k) {
    final int prediction = IntStream.range(0, this.data.dim()).mapToObj(i -> i).sorted(Comparator.comparing(i -> this.data.getData()[i])).findFirst().get();
    return k == prediction ? 0 : 1;
  }

  public final double errRms(final int k) {
    return this.data.rms(ideal(k));
  }

  public abstract void feedback(final NDArray data, DeltaBuffer buffer);

  public final NDArray ideal(final int k) {
    final NDArray delta = new NDArray(this.data.getDims());
    Arrays.parallelSetAll(delta.getData(), i -> i == k ? 1. : 0.);
    return delta;
  }

  public abstract boolean isAlive();

}
