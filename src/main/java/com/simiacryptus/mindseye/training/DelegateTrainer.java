package com.simiacryptus.mindseye.training;

import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.util.ml.Tensor;

public abstract class DelegateTrainer<T extends TrainingComponent> implements TrainingComponent {
    protected final T inner;
    private boolean verbose = false;

    public DelegateTrainer(T inner) {
        super();
        this.inner = inner;
    }

    @Override
    public Tensor[][] getTrainingData() {
      return this.inner.getTrainingData();
    }

    @Override
    public double getError() {
      return this.inner.getError();
    }

    @Override
    public DAGNetwork getNet() {
      return this.inner.getNet();
    }

    public boolean isVerbose() {
      return this.verbose;
    }

    @Override
    public void reset() {
        this.inner.reset();
    }

    @Override
    public TrainingComponent setData(final Tensor[][] data) {
      return this.inner.setData(data);
    }

    public TrainingComponent setVerbose(final boolean verbose) {
      this.verbose = verbose;
      return this;
    }
}
