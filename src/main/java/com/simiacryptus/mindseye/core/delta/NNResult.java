package com.simiacryptus.mindseye.core.delta;

import com.simiacryptus.mindseye.core.NDArray;

public abstract class NNResult {

  public final NDArray[] data;

  public NNResult(final NDArray... data) {
    super();
    this.data = data;
  }

  public final void accumulate(DeltaSet buffer) {
    accumulate(buffer, java.util.stream.IntStream.range(0, data.length).mapToObj(i->new NDArray(data[0].getDims()).fill(()->1.)).toArray(i->new NDArray[i]));
  }

  public abstract void accumulate(DeltaSet buffer, final NDArray[] data);

  public abstract boolean isAlive();

}
