package com.simiacryptus.mindseye.core.delta;

import com.simiacryptus.mindseye.core.NDArray;

public abstract class NNResult {

  public static NNResult add(final NNResult a, final NNResult b) {
    return new NNResult(a.data.add(b.data)) {

      @Override
      public void feedback(final NDArray data, final DeltaSet buffer) {
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
    return new NNResult(eval.data.scale(d)) {

      @Override
      public void feedback(final NDArray data, final DeltaSet buffer) {
        eval.feedback(data.scale(d), buffer);
      }

      @Override
      public boolean isAlive() {
        return eval.isAlive();
      }
    };
  }

  public final NDArray data;

  public NNResult(final NDArray data) {
    super();
    this.data = data;
  }

  public abstract void feedback(final NDArray data, DeltaSet buffer);

  public abstract boolean isAlive();

}
