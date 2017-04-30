package com.simiacryptus.mindseye.training;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.util.ml.Tensor;

public abstract class StochasticTrainer extends TrainerBase {
    private long hash = Util.R.get().nextLong();
    private int trainingSize = Integer.MAX_VALUE;

    public Tensor[][] selectTrainingData() {
      final Tensor[][] rawData = getTrainingData();
      assert 0 < rawData.length;
      assert 0 < getTrainingSize();
      return java.util.Arrays.stream(rawData).parallel() //
          .sorted(java.util.Comparator.comparingLong(y -> System.identityHashCode(y) ^ this.hash)) //
          .limit(getTrainingSize()) //
          .toArray(i -> new Tensor[i][]);
    }

    public int getTrainingSize() {
      return this.trainingSize;
    }

    public TrainerBase setTrainingSize(final int trainingSize) {
      updateHash();
      this.trainingSize = trainingSize;
      return this;
    }

    public void updateHash() {
      this.hash = Util.R.get().nextLong();
    }
}
