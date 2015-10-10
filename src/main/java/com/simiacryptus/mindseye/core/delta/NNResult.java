package com.simiacryptus.mindseye.core.delta;

import com.simiacryptus.mindseye.core.NDArray;

public abstract class NNResult {

  public static NNResult add(final NNResult a, final NNResult b) {
    return new NNResult(a.data.add(b.data)) {

      @Override
      public void accumulate(final DeltaSet buffer, final NDArray data) {
        a.accumulate(buffer, data);
        b.accumulate(buffer, data);
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
      public void accumulate(final DeltaSet buffer, final NDArray data) {
        eval.accumulate(buffer, data.scale(d));
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

  public final void accumulate(DeltaSet buffer) {
    accumulate(buffer, new NDArray(data.getDims()).fill(()->1.));
  }

  public abstract void accumulate(DeltaSet buffer, final NDArray data);

  public abstract boolean isAlive();

}
