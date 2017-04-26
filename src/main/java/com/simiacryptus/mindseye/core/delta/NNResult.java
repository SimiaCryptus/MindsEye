package com.simiacryptus.mindseye.core.delta;

import com.simiacryptus.util.ml.Tensor;

public abstract class NNResult {

  public final Tensor[] data;

  public NNResult(final Tensor... data) {
    super();
    this.data = data;
  }

  public final void accumulate(DeltaSet buffer) {
    accumulate(buffer, java.util.stream.IntStream.range(0, data.length).mapToObj(i->new Tensor(data[0].getDims()).fill(()->1.)).toArray(i->new Tensor[i]));
  }

  public abstract void accumulate(DeltaSet buffer, final Tensor[] data);

  public abstract boolean isAlive();

}
