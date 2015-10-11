package com.simiacryptus.mindseye.core.delta;

import com.simiacryptus.mindseye.core.NDArray;

public abstract class NNResult {

  public static NNResult add(final NNResult a, final NNResult b) {
    NDArray[] sums = new NDArray[a.data.length];
    for(int i=0;i<sums.length;i++) {
      sums[i] = a.data[i].add(b.data[i]);
    }
    return new NNResult(sums) {

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
    NDArray[] sums = new NDArray[eval.data.length];
    for(int i=0;i<sums.length;i++) {
      sums[i] = eval.data[i].scale(d);
    }
    return new NNResult(sums) {

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

  public final NDArray[] data;

  public NNResult(final NDArray... data) {
    super();
    this.data = data;
  }

  public final void accumulate(DeltaSet buffer) {
    accumulate(buffer, new NDArray(data[0].getDims()).fill(()->1.));
  }

  public abstract void accumulate(DeltaSet buffer, final NDArray data);

  public abstract boolean isAlive();

}
