package com.simiacryptus.mindseye.deltas;

import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.IntStream;

import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.dag.EvaluationContext;

public abstract class NNResult {

  public static NNResult add(final NNResult a, final NNResult b) {
    assert a.evaluationContext == b.evaluationContext;
    return new NNResult(a.evaluationContext, a.data.add(b.data)) {

      @Override
      public void feedback(final NDArray data, final DeltaBuffer buffer) {
        a.feedback(data, buffer);
        b.feedback(data, buffer);
      }

      @Override
      public boolean isAlive() {
        return a.isAlive() || b.isAlive();
      }
    };
  }

  public static NNResult scale(final NNResult eval, final double d) {
    return new NNResult(eval.evaluationContext, eval.data.scale(d)) {

      @Override
      public void feedback(final NDArray data, final DeltaBuffer buffer) {
        eval.feedback(data.scale(d), buffer);
      }

      @Override
      public boolean isAlive() {
        return eval.isAlive();
      }
    };
  }

  public final NDArray data;

  public final EvaluationContext evaluationContext;

  public NNResult(final EvaluationContext evaluationContext, final NDArray data) {
    super();
    this.data = data;
    this.evaluationContext = evaluationContext;
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
