package com.simiacryptus.mindseye.net;

import com.simiacryptus.util.ml.Tensor;

import java.util.Arrays;
import java.util.stream.IntStream;

public abstract class NNResult {

  public final Tensor[] data;

  public NNResult(final Tensor... data) {
    super();
    this.data = data;
  }

  public static NNResult[] singleResultArray(Tensor[][] input) {
    return Arrays.stream(input).map((Tensor[] x) -> new NNLayer.ConstNNResult(x)).toArray(i -> new NNResult[i]);
  }

  public static NNResult[] singleResultArray(Tensor[] input) {
    return Arrays.stream(input).map((Tensor x) -> new NNLayer.ConstNNResult(x)).toArray(i -> new NNResult[i]);
  }

  public static NNResult[] batchResultArray(Tensor[][] batchData) {
    return IntStream.range(0, batchData[0].length).mapToObj(inputIndex->{
      Tensor[] inputBatch = IntStream.range(0, batchData.length).mapToObj(trainingExampleId->
              batchData[trainingExampleId][inputIndex]).toArray(i->new Tensor[i]);
      return new NNLayer.ConstNNResult(inputBatch);
    }).toArray(x->new NNResult[x]);
  }

  public final void accumulate(DeltaSet buffer) {
    Tensor[] defaultVector = IntStream.range(0, this.data.length).mapToObj(i -> {
      assert(Arrays.equals(this.data[i].getDims(), new int[]{1}));
      return new Tensor(this.data[i].getDims()).fill(() -> 1.);
    }).toArray(i -> new Tensor[i]);
    accumulate(buffer, defaultVector);
  }

  public abstract void accumulate(DeltaSet buffer, final Tensor[] data);

  public abstract boolean isAlive();

}
