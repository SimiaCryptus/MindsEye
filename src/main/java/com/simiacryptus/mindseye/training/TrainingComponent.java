package com.simiacryptus.mindseye.training;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;

public interface TrainingComponent {

  public static class TrainingStep {
    private boolean changed;
    private double startError;
    double testError;

    public TrainingStep(final double startError, final double endError, final boolean changed) {
      super();
      setStartError(startError);
      this.testError = endError;
      setChanged(changed);
    }

    public double finalError() {
      return isChanged() ? testError : getStartError();
    }

    public double getStartError() {
      return startError;
    }

    public double improvement() {
      return getStartError() - testError;
    }

    public boolean isChanged() {
      return changed;
    }

    public void setChanged(final boolean changed) {
      this.changed = changed;
    }

    public void setStartError(final double startError) {
      this.startError = startError;
    }
  }

  NDArray[][] getData();

  double getError();

  DAGNetwork getNet();

  void reset();

  TrainingComponent setData(NDArray[][] data);

  TrainingStep step(TrainingContext trainingContext) throws TerminationCondition;

}
